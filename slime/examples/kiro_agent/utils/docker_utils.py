import docker
import logging
import os
import subprocess
import time
import uuid
import tempfile
import tarfile
from pathlib import PurePosixPath, Path
from typing import Any, Dict, Tuple, Optional

STRANDS_ROOT = Path(__file__).parent.parent
MAX_DOCKER_CONCURRENCY = 4
BRAZIL_USER = 'p4admin'

def set_volume_permissions(container_id, volume_path: Path):
    # Make sure we can read the volume
    # Docker is running as root, we may be running as augment.
    my_uid = os.getuid()
    my_gid = os.getgid()
    logging.info(f"Fixing permissions for {volume_path} to {my_uid}:{my_gid}")
    env = os.environ.copy()
    try:
        subprocess.check_call(
            [
                "chmod",
                "a+rx",
            ]
            + [p.as_posix() for p in volume_path.parents],
            env=env,
        )
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to chmod {volume_path}: {e}")
        raise
    # Change the owner to the current user
    try:
        container_out = subprocess.check_output(
            [
                "chown",
                "-R",
                f"{my_uid}:{my_gid}",
                volume_path.as_posix(),
            ],
            env=env,
            text=True,
            errors="backslashreplace",
        )
        logging.debug(container_out)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to chown {volume_path}: {e}")
        raise


def remove_container_image(image_name: str) -> None:
    """Remove a docker image."""
    try:
        client = docker.from_env(timeout=120)
        client.images.remove(image=image_name, force=True)
        logging.info(f"Removed image {image_name}")
    except docker.errors.APIError as e:  # type: ignore
        logging.warning(f"Failed to remove image {image_name}: {e}")


def stop_container(container_id: str, remove_image: str = "") -> None:
    """Stop a docker container for the issue."""
    container = None
    try:
        client = docker.from_env(timeout=120)
        container = client.containers.get(container_id)
    except Exception as e:
        logging.info(f"Container {container_id} not found: {e}")

    if container:
        try:
            logging.info(f"Stopping container {container_id}")
            container.stop()
            logging.info(f"Stopped container {container_id}")
        except docker.errors.NotFound as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")
        except docker.errors.APIError as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")
        try:
            logging.info(f"Removing container {container_id}")
            container.remove()
            time.sleep(10)
            logging.info(f"Removed container {container_id}")
        except docker.errors.NotFound as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")
        except docker.errors.APIError as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")

    if remove_image:
        # Add a small delay to ensure container removal is complete
        time.sleep(5)
        remove_container_image(remove_image)


def start_aswe_container(workspace: Path, problem_id: str, mode: str,  semaphore: Any, local_repo_path: str) -> str:
    """Start a docker container for the issue."""
    stop_container(f"asweb.strands.{problem_id}")
    
    # If local_repo_path provided, it means we are copying in workspace into base image.
    # Otherwise, we are using the instance-level execution image
    if local_repo_path:
        image_name = "strands_v0.1.4_base_image"
    else:
        image_name = get_execution_image(problem_id)

    logging.info(f"Starting container for {problem_id}")
    client = docker.from_env(timeout=120)
    logging.info(f"Pulling image {image_name}")

    if local_repo_path:
        # create a kiro_specs folder under workspace/problem_id
        subprocess.check_output(
            ["cp","-r",os.path.join(local_repo_path,"."), (workspace/problem_id)],
            text=True,
        )
        logging.info(f"Copy from {local_repo_path} to tmp workspace {(workspace/problem_id)}")

    logging.info(f"Running docker run for {image_name} in {workspace}")
    with semaphore:
        logging.info(f"Starting run for {image_name}")
        if local_repo_path:
            container = client.containers.run(
                name=f"asweb.strands.{problem_id}_{uuid.uuid4().hex[:8]}",
                image=image_name,
                detach=True,
                volumes=[f"{(workspace/problem_id)}:/testbed"],
                command="bash -c 'cd ./testbed && git init && git config --global user.email a && git config --global user.name a && git config --global --add safe.directory /testbed &&  git add . && git commit --allow-empty -am kiro-on-strands && sleep 7200'", # Time out and die, eventually, if we are interrupted
            )
            if mode == "specs":
                kiro_spec_path = workspace / problem_id / ".kiro/specs"
                kiro_spec_path.mkdir(parents=True, exist_ok=True)
            
            if mode == "steering":
                kiro_steering_file_path = workspace / problem_id / ".kiro/steering"
                kiro_steering_file_path.mkdir(parents=True, exist_ok=True)
        else:
            container = client.containers.run(
                name=f"asweb.strands.{problem_id}_{uuid.uuid4().hex[:8]}",
                image=image_name,
                detach=True,
                command="bash -c 'sleep 7200'"
            )
        logging.info(f"Finished startup for {image_name}")
    # Give it a second to start
    time.sleep(10)
    container_id = container.id
    assert container_id is not None
    
    # Fix permissions for .kiro directory if it exists
    kiro_path = workspace / problem_id / ".kiro"
    if kiro_path.exists():
        set_volume_permissions(container_id, kiro_path)
    
    logging.info(f"Started {container_id} for {problem_id}")
    return container_id

def start_aswe_container_public(workspace: Path, problem_id: str, base_commit:str, mode: str,  semaphore: Any, local_repo_path: str) -> str:
    """Start a docker container for the issue."""
    stop_container(f"asweb.strands.{problem_id}")
    if local_repo_path:
        image_name = "strands_v0.1.4_base_image"
    else:
        image_name = get_execution_image(problem_id)

    logging.info(f"Starting container for {problem_id}")
    client = docker.from_env(timeout=120)

    if local_repo_path:
        subprocess.check_output(
            ["cp","-r", os.path.join(local_repo_path,"."), (workspace/problem_id)],
            text=True,
        )
        logging.info(f"Copy from {local_repo_path} to tmp workspace {(workspace/problem_id)}")

    logging.info(f"Running docker run for {image_name} in {workspace}")
    with semaphore:
        logging.info(f"Starting run for {image_name}")
        if local_repo_path:
            container = client.containers.run(
                    name=f"asweb.strands.{problem_id}_{uuid.uuid4().hex[:8]}",
                    image=image_name,
                    detach=True,
                    volumes=[f"{(workspace/problem_id)}:/testbed"],
                    command=f"bash -c 'cd ./testbed && git config --global user.email a && git config --global user.name a && git config --global --add safe.directory /testbed && git reset --hard && git checkout {base_commit} && sleep 7200'"
            )
            if mode == "specs":
                kiro_spec_path = workspace / problem_id / ".kiro/specs"
                kiro_spec_path.mkdir(parents=True, exist_ok=True)
            
            if mode == "steering":
                kiro_steering_file_path = workspace / problem_id / ".kiro/steering"
                kiro_steering_file_path.mkdir(parents=True, exist_ok=True)
        else:
            container = client.containers.run(
                name=f"asweb.strands.{problem_id}_{uuid.uuid4().hex[:8]}",
                image=image_name,
                detach=True,
                command="bash -c 'sleep 7200'"
            )
        logging.info(f"Finished startup for {image_name}")
    # Give it a second to start
    time.sleep(10)
    container_id = container.id
    assert container_id is not None
    
    # Fix permissions for .kiro directory if it exists
    kiro_path = workspace / problem_id / ".kiro"
    if kiro_path.exists():
        set_volume_permissions(container_id, kiro_path)
    
    logging.info(f"Started {container_id} for {problem_id}")

    return container_id


def setup_aswe_workspace(workspace: Path, problem: dict, mode: str, lock: Any, semaphore: Any, repo_base: str, dataset_type: str = "aswe"
) -> Tuple[Dict[str, str], str]:
    """Setup the workspace for the agent."""
    env: Dict[str, str] = os.environ.copy()
    if dataset_type in ['aswe']:
        local_repo = os.path.join(repo_base, problem['repo'], problem['base_commit'])
        container_id = start_aswe_container(workspace, problem['instance_id'], mode, semaphore, local_repo)
    else:
        local_repo =  os.path.join(repo_base, problem['repo'])
        container_id = start_aswe_container_public(workspace, problem['instance_id'], problem['base_commit'], mode, semaphore, local_repo)
    
    return env, container_id


def wrap_filepath_for_docker(path: str, local_workspace: str, container_workspace: Optional[str] = None) -> Path:
    """Given a path, possibly in a container workspace, return the absolute local path."""
    path = Path(path)
    root = Path(local_workspace)
    if container_workspace:
        container_workspace = Path(container_workspace)
    if not path.is_absolute():
        return root / path
    if container_workspace and path.is_relative_to(container_workspace):
        return root / path.relative_to(container_workspace)
    return path

def get_absolute_docker_path(path: str, container_workspace: str) -> str:
    path = Path(path)
    container_workspace = Path(container_workspace)
    if not path.is_absolute():
        return container_workspace / path
    if path.is_relative_to(container_workspace):
        return container_workspace / path.relative_to(container_workspace)
    return str(path)


def wrap_command_for_docker(command: str, container: str) -> str:
    """Wrap a command for execution in a Docker container.

    Args:
        command: Command to execute in container

    Returns:
        Docker exec command string
    """
    docker_parts = ["docker", "exec"]

    # Need to run brazil-build as p4admin
    if "brazil-build" in command:
        docker_parts.extend(["--user", BRAZIL_USER])

    docker_parts.append(container)

    # For docker exec, we use the shell to handle command properly
    escaped_cmd = command.replace('"', '\\"')
    docker_parts.extend(["/bin/bash", "-l", "-c", f'"{escaped_cmd}"'])

    return " ".join(docker_parts)


def get_execution_image(problem_id: str) -> str:
    repo = problem_id.split('_')[0].lower()
    commit_id = problem_id.split('_')[-1][:10]
    image_name = f"975050351917.dkr.ecr.us-west-2.amazonaws.com/aswe-v2-java/eval-env:{repo}--{commit_id}"
    return image_name


# ============================================================================
# CGS (Context Gathering SFT) Container Support Functions
# ============================================================================

CGS_DEFAULT_IMAGE = "975050351917.dkr.ecr.us-east-2.amazonaws.com/q-codegen:skgouda-kiro-strands-cgs-eval-20251224"

def start_cgs_container(workspace: Path, problem_id: str, semaphore: Any, local_repo_path: str, cgs_image: str = None) -> str:
    """Start a docker container for CGS (Context Gathering SFT) task.
    
    Args:
        workspace: Base workspace path
        problem_id: Instance ID for the problem
        semaphore: Unused, kept for API compatibility
        local_repo_path: Path to the local repository to mount
        cgs_image: Docker image to use (defaults to CGS_DEFAULT_IMAGE)
        
    Returns:
        Container ID string
    """
    container_name = f"cgs.strands.{problem_id}"
    stop_container(container_name)
    
    image_name = cgs_image or CGS_DEFAULT_IMAGE
    
    logging.info(f"Starting CGS container for {problem_id}")
    client = docker.from_env(timeout=120)
    
    # Copy repo to workspace
    workspace_instance = workspace / problem_id
    if local_repo_path and os.path.exists(local_repo_path):
        subprocess.check_output(
            ["cp", "-r", os.path.join(local_repo_path, "."), str(workspace_instance)],
            text=True,
        )
        logging.info(f"Copied from {local_repo_path} to workspace {workspace_instance}")
    
    logging.info(f"Running docker run for {image_name} in {workspace}")
    logging.info(f"Starting run for {image_name}")
    container = client.containers.run(
        name=f"{container_name}_{uuid.uuid4().hex[:8]}",
        image=image_name,
        detach=True,
        volumes=[f"{workspace_instance}:/testbed"],
        command="bash -c 'cd /testbed && git config --global user.email a && git config --global user.name a && git config --global --add safe.directory /testbed && sleep 7200'",
    )
    logging.info(f"Finished startup for {image_name}")
    
    time.sleep(5)
    container_id = container.id
    assert container_id is not None
    
    # Verify files exist in /testbed
    try:
        exit_code, output = container.exec_run("ls -la /testbed", demux=True)
        stdout = output[0].decode() if output[0] else ""
        file_count = len([l for l in stdout.split('\n') if l.strip() and not l.startswith('total')])
        logging.info(f"CGS container {problem_id}: /testbed has {file_count} entries")
        if file_count <= 2:  # Only . and ..
            logging.warning(f"CGS container {problem_id}: /testbed appears empty! Contents: {stdout[:200]}")
    except Exception as e:
        logging.warning(f"CGS container {problem_id}: Failed to verify /testbed contents: {e}")
    
    logging.info(f"Started CGS container {container_id} for {problem_id}")
    return container_id


def setup_cgs_workspace(workspace: Path, problem: dict, semaphore: Any, repo_base: str, cgs_image: str = None) -> Tuple[Dict[str, str], str]:
    """Setup the workspace for CGS (Context Gathering SFT) task.
    
    Args:
        workspace: Base workspace path
        problem: Problem dict with instance_id, repo, base_commit
        semaphore: Semaphore for controlling Docker concurrency
        repo_base: Base path where repos are stored
        cgs_image: Docker image to use
        
    Returns:
        Tuple of (environment dict, container_id)
    """
    env: Dict[str, str] = os.environ.copy()
    
    # CGS repo path format: repo_base/repo_name (repo includes org/name/git_repo_name)
    # e.g., /mnt/s3-mount/zhenglee/data/synthetic_trajectory/repos/5afe/safe-react/git_repo_safe-react
    local_repo = os.path.join(repo_base, problem['repo'])
    
    container_id = start_cgs_container(
        workspace, 
        problem['instance_id'], 
        semaphore, 
        local_repo,
        cgs_image
    )
    
    return env, container_id

def apply_patch_in_container(container_id: str, repo_name: str, patch: str):
    container = None
    try:
        client = docker.from_env(timeout=120)
        container = client.containers.get(container_id)
    except Exception as e:
        logging.info(f"Container {container_id} not found: {e}")

    repo_path_in_container = f"/home/p4admin/workspace/src/{repo_name}"
    repo_path = PurePosixPath(repo_path_in_container)
    patch_file = "patch.patch"
    patch_in_container = PurePosixPath(repo_path / patch_file)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        patch_path = temp_dir / patch_file
        with open(patch_path, "w") as f:
            f.write(patch)

        src = patch_path
        dst = patch_in_container
        # temporary tar file
        tar_path = src.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(
                src, arcname=dst.name
            )  # use destination name, so after `put_archive`, name is correct

        # get bytes for put_archive cmd
        with open(tar_path, "rb") as tar_file:
            data = tar_file.read()

        # Make directory if necessary
        container.exec_run(f"mkdir -p {dst.parent}")

        # Send tar file to container and extract
        container.put_archive(os.path.dirname(dst), data)

        # clean up in locally and in container
        tar_path.unlink()

def get_patch_from_container(container_id: str, repo_name: str, base_commit: str) -> str:
    container = None
    try:
        client = docker.from_env(timeout=120)
        container = client.containers.get(container_id)
    except Exception as e:
        logging.info(f"Container {container_id} not found: {e}")

    repo_path_in_container = f"/home/p4admin/workspace/src/{repo_name}"

    # Run the git diff command
    # Don't need 'git apply -R patch.patch' since we can remove the test patches manually later
    cmd = ["/bin/bash", "-l", "-c", f"git --git-dir={repo_path_in_container}/.git --work-tree={repo_path_in_container} diff {base_commit}"]
    exit_code, output = container.exec_run(
        cmd,
        workdir=repo_path_in_container,
        stdout=True,
        stderr=True,
        demux=True,  # demux to separate stdout and stderr
    )
    stdout, stderr = output

    if exit_code != 0:
        raise RuntimeError(f"Git diff failed in {repo_path_in_container}: {stderr.decode() if stderr else 'Unknown error'}")

    # This is the raw diff content
    diff = stdout.decode() if stdout else ""
    return diff


def pull_aswe_image():
    """Pull the ASWE evaluation image and load strands base image if available."""
    image_name = "nikolaik/python-nodejs:python3.12-nodejs22"
    client = docker.from_env()
    client.images.pull(image_name)
    
    # Try to load strands base image from tar file if it exists
    strands_image_tar = os.environ.get("STRANDS_IMAGE_TAR", "/mnt_private/wdimmy/AmazonSWEBench-V2/strands_v0.1.4_base_image.tar")
    if os.path.exists(strands_image_tar):
        logging.info(f"Loading strands base image from {strands_image_tar}")
        try:
            result = subprocess.run(
                ["docker", "load", "-i", strands_image_tar],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                logging.info(f"Successfully loaded strands base image: {result.stdout.strip()}")
            else:
                logging.warning(f"Failed to load strands base image: {result.stderr}")
        except Exception as e:
            logging.warning(f"Error loading strands base image: {e}")
    else:
        logging.warning(f"Strands image tar not found at {strands_image_tar}. Set STRANDS_IMAGE_TAR environment variable if needed.")


# ============================================================================
# SWE-Style Container Support Functions
# ============================================================================

def start_public_container_workspace(instance_id: str, semaphore: Any) -> str:
    """Start a docker container for public SWE task instance.
    
    Args:
        instance_id: Instance ID for the problem
        semaphore: Unused, kept for API compatibility
    """
    container_name = f"swe.strands.{instance_id}_{uuid.uuid4().hex[:8]}"
    stop_container(container_name)
    
    image_name = f"{instance_id}:latest"
    
    logging.info(f"Starting public SWE container for {instance_id}")
    client = docker.from_env()
    
    logging.info(f"Starting run for {image_name}")
    container = client.containers.run(
        name=container_name,
        tty=True,
        image=image_name,
        detach=True,
        command='tail -f /dev/null',
        network_mode="none",
        mem_limit="10g",
        memswap_limit="20g",
        entrypoint="",
    )
    logging.info(f"Finished startup for {image_name}")
    
    # Give it a second to start
    time.sleep(2)
    
    # Ensure ripgrep is installed for grepSearch/fileSearch tools
    ensure_ripgrep_installed(container)
    
    container_id = container.id
    assert container_id is not None
    
    logging.info(f"Started {container_id} for {instance_id}")
    return container_id


def setup_public_container_workspace(problem: dict, lock: Any, semaphore: Any) -> Tuple[Dict[str, str], str]:
    """Setup the workspace for public SWE task agent execution."""
    env: Dict[str, str] = os.environ.copy()
    
    instance_id = problem['instance_id']
    env["INSTANCE_ID"] = instance_id
    
    container_id = start_public_container_workspace(instance_id, semaphore)
    
    return env, container_id


def get_patch_from_swe_container(container_id: str, workdir: str) -> str:
    """Extract git diff patch from a SWE container's workspace.
    
    Args:
        container_id: Docker container ID
        workdir: Path to the git repo inside the container
        
    Returns:
        Git diff as a string
    """
    container = None
    try:
        client = docker.from_env(timeout=120)
        container = client.containers.get(container_id)
    except Exception as e:
        logging.error(f"Container {container_id} not found: {e}")
        raise

    # Run git diff command in the container
    # First add all files (including untracked) so they show up in the diff
    cmd = [
        "/bin/bash", "-c",
        f"cd {workdir} && git add -A && git --no-pager diff -U5 --no-color --cached HEAD"
    ]
    exit_code, output = container.exec_run(
        cmd,
        workdir=workdir,
        stdout=True,
        stderr=True,
        demux=True,
    )
    stdout, stderr = output

    if exit_code != 0:
        error_msg = stderr.decode() if stderr else 'Unknown error'
        logging.error(f"Git diff failed in {workdir}: {error_msg}")
        raise RuntimeError(f"Git diff failed in {workdir}: {error_msg}")

    diff = stdout.decode() if stdout else ""
    return diff


def preload_swe_docker_images(instance_ids: list, source_path: str):
    """Preload Docker images for SWE instances from tar.gz files.
    
    Args:
        instance_ids: List of instance IDs to load images for
        source_path: Directory containing tar.gz image files
        
    Returns:
        Tuple of (loaded_images, failed_images)
    """
    try:
        docker_client = docker.from_env(timeout=600)
    except Exception as e:
        logging.warning(f"Docker client initialization failed: {e}")
        return [], instance_ids

    start_time = time.time()
    loaded_images = []
    failed_images = []

    for instance_id in instance_ids:
        try:
            image_file = Path(source_path) / f"{instance_id}.tar.gz"
            if not image_file.exists():
                logging.error(f"Image file not found: {image_file}")
                failed_images.append(instance_id)
                continue

            # Load image from file
            logging.info(f"Loading image for {instance_id} from {image_file}")
            with open(image_file, 'rb') as f:
                loaded_images_info = docker_client.images.load(f.read())

            # Tag the loaded image
            for img_info in loaded_images_info:
                if hasattr(img_info, 'id'):
                    image = docker_client.images.get(img_info.id)
                    image.tag(f"{instance_id}:latest")
                    logging.info(f"Successfully tagged image: {instance_id}")
                    loaded_images.append(instance_id)
                    break

        except Exception as e:
            logging.error(f"Failed to load image {instance_id}: {e}")
            failed_images.append(instance_id)

    total_time = time.time() - start_time
    logging.info(f"Image loading completed in {total_time:.2f}s. Loaded: {len(loaded_images)}, Failed: {len(failed_images)}")
    
    return loaded_images, failed_images


def cleanup_swe_docker_resources():
    """Clean up SWE-related Docker resources."""
    try:
        docker_client = docker.from_env(timeout=600)
    except Exception as e:
        logging.warning(f"Docker client initialization failed: {e}")
        return

    containers_removed = 0
    images_removed = 0
    
    try:
        # Remove SWE containers
        all_containers = docker_client.containers.list(all=True)
        for container in all_containers:
            if container.name and "swe.strands." in container.name:
                try:
                    container.remove(force=True)
                    containers_removed += 1
                except Exception as e:
                    logging.warning(f"Could not remove container {container.name}: {e}")
        
        logging.info(f"Removed {containers_removed} SWE containers")
        
        # Remove SWE images (those tagged with instance IDs)
        images = docker_client.images.list()
        for image in images:
            try:
                # Check if image has tags that look like instance IDs
                if image.tags:
                    for tag in image.tags:
                        if ":" in tag and tag.endswith(":latest"):
                            instance_part = tag.split(":")[0]
                            # Simple heuristic: if it looks like an instance ID
                            if "_" in instance_part or instance_part.isalnum():
                                docker_client.images.remove(image.id, force=True)
                                images_removed += 1
                                break
            except Exception as e:
                logging.warning(f"Could not remove image {image.id}: {e}")

        logging.info(f"Removed {images_removed} SWE images")
        
    except Exception as e:
        logging.error(f"Error in cleanup_swe_docker_resources: {str(e)}")


def ensure_ripgrep_installed(container) -> bool:
    """Install ripgrep if not available in container.
    
    ripgrep (rg) is required for grepSearch and fileSearch tools.
    This function attempts to copy it from host or install via package managers.
    
    Args:
        container: Docker container object
        
    Returns:
        True if ripgrep is available (already installed or successfully installed),
        False if installation failed
    """
    # Check if rg already exists
    exit_code, _ = container.exec_run(["which", "rg"])
    if exit_code == 0:
        return True
    
    logging.info("ripgrep (rg) not found in container, attempting to install...")
    
    # First try: copy from host (works even without network)
    import subprocess
    import tarfile
    import io
    
    host_rg = subprocess.run(["which", "rg"], capture_output=True, text=True)
    if host_rg.returncode == 0:
        rg_path = host_rg.stdout.strip()
        try:
            # Create tar archive with rg binary
            tarstream = io.BytesIO()
            with tarfile.open(fileobj=tarstream, mode='w') as tar:
                tar.add(rg_path, arcname='rg')
            tarstream.seek(0)
            
            # Copy to container
            container.put_archive('/usr/local/bin', tarstream)
            container.exec_run(["chmod", "+x", "/usr/local/bin/rg"])
            
            # Verify
            verify_code, _ = container.exec_run(["which", "rg"])
            if verify_code == 0:
                logging.info("ripgrep copied from host successfully")
                return True
        except Exception as e:
            logging.warning(f"Failed to copy ripgrep from host: {e}")
    
    # Fallback: try package managers (requires network)
    install_commands = [
        "apt-get update -qq && apt-get install -y -qq ripgrep 2>/dev/null",
        "apk add --no-cache ripgrep 2>/dev/null",
        "yum install -y ripgrep 2>/dev/null || dnf install -y ripgrep 2>/dev/null",
    ]
    
    for cmd in install_commands:
        exit_code, output = container.exec_run(["sh", "-c", cmd])
        if exit_code == 0:
            verify_code, _ = container.exec_run(["which", "rg"])
            if verify_code == 0:
                logging.info("ripgrep installed successfully")
                return True
    
    logging.warning("Failed to install ripgrep in container")
    return False