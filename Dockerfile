# Build:  docker buildx build --platform linux/amd64 -t enzyme-ec:submission --load .
# Run  :  docker run --rm --platform linux/amd64 --network none -v $PWD/io:/io enzyme-ec:submission \
#             python /app/predict.py --input /io/in.parquet --output /io/out.jsonl
FROM --platform=linux/amd64 python:3.11-slim-bookworm
RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user . .
# bin/diamond is the official DIAMOND v2.1.10 linux-x86_64 release binary (needs only glibc, libz, libm, libpthread)
RUN chmod 755 /app/bin/diamond && /app/bin/diamond version
USER user
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/predict.py", "--help"]
