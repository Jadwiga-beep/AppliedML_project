#!/bin/bash

IMAGE_FILE=$1

if [ ! -f "$IMAGE_FILE" ]; then
  URL=$IMAGE_FILE
  IMAGE_FILE=/tmp/test_image.bin
  if ! curl --output $IMAGE_FILE --silent --head --fail "$URL"; then
    echo "ERROR: $URL is not a valid file nor an image URL"
    exit 1
  fi
  curl -s -o $IMAGE_FILE --silent "$URL"
fi

curl -X POST http://127.0.0.1:8000/predict -F "file=@$IMAGE_FILE" | jq

[ -n "$URL" ] && rm -f "$IMAGE_FILE"