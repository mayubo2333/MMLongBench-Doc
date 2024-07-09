#!/bin/bash

mkdir -p ./results
mkdir -p ./tmp

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error. You do not set the OPENAI_API_KEY. It is necessary for the answer extractor for evaluation."
  exit 1
fi

if [[ "$MODEL_NAME" == *"gemini"* ]] && [ -z "$GOOGLE_API_KEY" ]; then
  echo "Error. You do not set the GOOGLE_API_KEY. It is necessary for Gemini series."
  exit 1
fi


if [[ "$MODEL_NAME" == *"gemini"* ]] || [[ "$MODEL_NAME" == *"gpt-4"* ]]; then
  python run_api.py --model_name $MODEL_NAME
else
  if [ -z "$CACHE_PATH" ]; then
    export CACHE_PATH="None"
  fi
  python run_lvlm.py --model_name $MODEL_NAME --model_cached_path $CACHE_PATH
fi