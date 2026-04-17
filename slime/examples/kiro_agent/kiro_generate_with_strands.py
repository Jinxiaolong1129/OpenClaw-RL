# ============================================================================
# Alternative: Strands-based Generation (Optional)
# ============================================================================

async def generate_with_strands(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    Alternative generate function using strands-sglang.
    
    This provides TITO (Token-In-Token-Out) tracking for exact token alignment.
    Requires: pip install strands-sglang
    
    Usage:
        --custom-generate-function-path examples.kiro_on_strands.kiro_generate:generate_with_strands
    """
    try:
        from strands import Agent, tool
        from strands_sglang import SGLangClient, SGLangModel
        from strands_sglang.tool_limiter import ToolIterationLimiter
    except ImportError:
        logger.warning("strands-sglang not installed, falling back to basic generate")
        return await generate(args, sample, sampling_params)
    
    assert not args.partial_rollout, "Partial rollout not supported"
    
    state = GenerateState(args)
    instance_id = sample.metadata.get("instance_id", f"instance_{sample.index}")
    rollout_idx = sample.metadata.get("rollout_idx", sample.index % args.n_samples_per_prompt)
    
    workspace = await setup_workspace(sample, rollout_idx)
    
    if not workspace.is_active:
        sample.status = Sample.Status.FAILED
        sample.response = "Error: Workspace setup failed"
        sample.tokens = []
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
        return sample
    
    try:
        # Create SGLang client and model
        client = SGLangClient.from_slime_args(args)
        model = SGLangModel(
            tokenizer=state.tokenizer,
            client=client,
            model_id=args.hf_checkpoint.split("/")[-1],
            params={k: sampling_params[k] for k in ["max_new_tokens", "temperature", "top_p"]},
        )
        
        # Define tools using strands @tool decorator
        @tool
        def execute_command(command: str) -> str:
            """Execute a shell command in the workspace."""
            return asyncio.get_event_loop().run_until_complete(
                docker_exec(workspace, command)
            )
        
        @tool
        def read_file(path: str) -> str:
            """Read the contents of a file."""
            return asyncio.get_event_loop().run_until_complete(
                docker_exec(workspace, f"cat '{path}'")
            )
        
        @tool
        def write_file(path: str, content: str) -> str:
            """Write content to a file."""
            return asyncio.get_event_loop().run_until_complete(
                docker_exec(workspace, f"cat > '{path}' << 'KIRO_EOF'\n{content}\nKIRO_EOF")
            )
        
        @tool
        def list_directory(path: str = ".") -> str:
            """List files in a directory."""
            return asyncio.get_event_loop().run_until_complete(
                docker_exec(workspace, f"ls -la '{path}'")
            )
        
        # Create agent with tools
        limiter = ToolIterationLimiter(max_iterations=CONFIG.max_iterations)
        agent = Agent(
            model=model,
            tools=[execute_command, read_file, write_file, list_directory],
            hooks=[limiter],
            callback_handler=None,
            system_prompt=CONFIG.system_prompt,
        )
        
        # Build prompt using instruction template from Kiro-on-Strands
        problem_statement = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
        file_tree = get_file_tree_from_container(workspace)
        
        prompt = f"""You are operating in a workspace with files and folders. Below is the known structure of the workspace.

{file_tree}

{CONFIG.instruction_template.format(pr_description=problem_statement, location=workspace.workdir)}"""
        
        try:
            await agent.invoke_async(prompt)
            sample.status = Sample.Status.COMPLETED
        except Exception as e:
            sample.status = Sample.Status.TRUNCATED
            logger.warning(f"Agent error: {type(e).__name__}: {e}")
        
        # Extract trajectory from token manager (TITO)
        tm = model.token_manager
        prompt_len = len(tm.segments[0])
        sample.tokens = tm.token_ids
        sample.loss_mask = tm.loss_mask[prompt_len:]
        sample.rollout_log_probs = tm.logprobs[prompt_len:]
        sample.response_length = len(sample.tokens) - prompt_len
        sample.response = model.tokenizer.decode(sample.tokens[prompt_len:], skip_special_tokens=False)
        
        # Extract patch
        patch = get_patch_from_container(workspace)
        
        # Store metadata
        sample.metadata["patch"] = patch
        sample.metadata["tool_iterations"] = limiter.iteration_count
        sample.metadata["tool_call_count"] = sum(
            1 for msg in agent.messages if getattr(msg, "role", None) == "tool"
        )
        
        model.reset()
        agent.cleanup()
    
    finally:
        await cleanup_workspace(workspace)
    
    return sample
