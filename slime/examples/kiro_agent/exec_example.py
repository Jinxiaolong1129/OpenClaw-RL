#!/usr/bin/env python3
"""
Query validate-service worker-0: health check, instance lookup,
and submit empty-patch validations for specific instance IDs.
"""

import concurrent.futures
import requests
import sys

SERVER_URL = "http://tiayuche-validate-service-cpu-worker-0:51429"
# http://tiayuche-validate-service-worker-0:51429/validate

INSTANCE_IDS = [
    "instance_spiffe__spire-70fe7a75c1d42b2d69b2c9b1a216595025d23d84",
    "instance_reown-com__appkit-36c8b88fd354503d2a4ca659870ca68f99a71248",
    "instance_facebook__react-native-7723c3132977c52b5620b23f56569c6a24d9367e",
    "instance_kyverno__kyverno-77b1294c38c6700af43d424cf07016a94f9202d1",
]


def validate_instance(instance_id: str, timeout: int = 900) -> dict:
    """Submit an empty patch for a single instance and return the result."""
    resp = requests.post(
        f"{SERVER_URL}/validate",
        json={"instance_id": instance_id, "patch": ""},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    # Health check
    print(f"Checking health at {SERVER_URL} ...")
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Status: {health['status']}, instances_loaded: {health['instances_loaded']}")
    except Exception as e:
        print(f"  Health check failed: {e}")
        sys.exit(1)

    # Submit all 4 validations concurrently with empty patches
    print(f"\nSubmitting empty-patch validation for {len(INSTANCE_IDS)} instances ...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(INSTANCE_IDS)) as pool:
        future_to_id = {
            pool.submit(validate_instance, iid): iid for iid in INSTANCE_IDS
        }

        for future in concurrent.futures.as_completed(future_to_id):
            iid = future_to_id[future]
            try:
                result = future.result()
                verdict = "PASS" if result.get("verdict") else "FAIL"
                error = result.get("error")
                exit_code = result.get("test_exit_code", -1)
                print(f"  [{verdict}] {iid}")
                print(f"         exit_code={exit_code}")
                if error:
                    print(f"         error: {error}")
            except Exception as e:
                print(f"  [ERROR] {iid}")
                print(f"         {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()