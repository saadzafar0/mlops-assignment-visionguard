#!/bin/bash
# Task 6.1: use only uniquely named temporary containers, tags, and a volume.
set -euo pipefail
toy_name="visionguard-docker-toy-$$"
toy_port="${DOCKER_TOY_PORT:-18081}"
toy_image="$toy_name:lab"
cleanup() {
  docker rm --force "$toy_name" >/dev/null 2>&1 || true
  docker volume rm "$toy_name" >/dev/null 2>&1 || true
  docker image rm "$toy_image" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf '23l-0829_A2_Ops Docker lab — %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
docker pull nginx:stable-alpine
docker image tag nginx:stable-alpine "$toy_image"
docker volume create "$toy_name"
docker run --detach --name "$toy_name" \
  --publish "127.0.0.1:$toy_port:80" \
  --mount "type=volume,source=$toy_name,target=/usr/share/nginx/html" "$toy_image"
curl --fail --silent --show-error --retry 15 --retry-all-errors --retry-delay 1 \
  "http://127.0.0.1:$toy_port/" >/dev/null
docker exec "$toy_name" sh -c \
  'printf "<h1>23l-0829_A2_Ops: named volume retained this page</h1>\n" > /usr/share/nginx/html/index.html'
curl --fail --silent --show-error "http://127.0.0.1:$toy_port/"
docker logs --tail 10 "$toy_name"
docker exec "$toy_name" sh -c 'id
ls -l /usr/share/nginx/html'
docker ps --filter "name=$toy_name"
docker image ls "$toy_image"
docker inspect --format '{{json .Mounts}}' "$toy_name"

docker stop "$toy_name"
docker rm "$toy_name"
docker run --detach --name "$toy_name" \
  --publish "127.0.0.1:$toy_port:80" \
  --mount "type=volume,source=$toy_name,target=/usr/share/nginx/html" "$toy_image"
page=$(curl --fail --silent --show-error --retry 15 --retry-all-errors --retry-delay 1 \
  "http://127.0.0.1:$toy_port/")
[[ "$page" == *'named volume retained this page'* ]]
printf '%s\n' "$page" 'The replacement container recovered the page from the named volume.'
docker rm --force "$toy_name"
docker volume rm "$toy_name"
docker image rm "$toy_image"
