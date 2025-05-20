import aiohttp
import asyncio
import time
import argparse
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=2000, n_features=100, noise=0.1)


model_path = "test-model"

# Example data matching your server's expected schema


def fit_payload(i):
    return {
        "x": X.tolist(),  # Replace with actual training data
        "model_name": model_path + f"{i}",
    }


async def post(session, payload):
    try:
        async with session.post(
            url="http://localhost:5123/predict",
            json=payload,
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(data)
            else:
                try:
                    error_data = await response.json()
                    print(f"Request failed with status: {response.status}")
                    print(f"Detail: {error_data.get('detail', 'No detail provided')}")
                except Exception:
                    print(f"Request failed with status: {response.status}, but no JSON detail returned.")
    except aiohttp.ClientError as e:
        print(f"Request failed due to client error: {e}")



async def main(num_requests):
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *[post(session, fit_payload(i)) for i in range(num_requests)]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AsyncFit")
    parser.add_argument(
        "--num_requests",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    ts = time.time()
    try:
        asyncio.run(main(args.num_requests))
    except Exception as e:
        print(f"Exception occurred: {e}")
