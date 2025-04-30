#!/usr/bin/env bash

LLM_COMMAND="llm"
MODEL_NAME="gemma3:12b-it-qat"
POSTS_DIR="."

# handle arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 \"Post Title\""
    echo "Example: $0 \"My Awesome New Blog Post\""
    exit 1
fi

POST_TITLE="$1"

# get date
CURRENT_DATE=$(date +%Y-%m-%d)
CURRENT_YEAR=$(date +%Y)
CURRENT_MONTH=$(date +%m)

# LLM prompt -- note that short-title is extracted from this later via regex
read -r -d '' LLM_PROMPT << EOM
\no_think
Today's date is: ${CURRENT_DATE}
The blog post title is: "${POST_TITLE}"

Generate YAML front matter for a blog post with the following structure.

Instructions:
1. Use the full, original title for the 'title' field, enclosed in single quotes.
2. Use the provided date '${CURRENT_DATE}' for the 'date' field.
3. Create a 'short-title': make the original title lowercase, replace all spaces with hyphens, remove any characters unsafe for URLs (keep only letters, numbers, hyphens), and keep it reasonably short (e.g., truncate logically if over ~4-5 words).
4. Construct the 'permalink' using the year (${CURRENT_YEAR}), month (${CURRENT_MONTH}), and the 'short-title': /blog/${CURRENT_YEAR}/${CURRENT_MONTH}/short-title/
5. Suggest a few (1-3) relevant tags like "blogging" or "galaxies" or "computer-vision" or "llms", depending on the post title.

Output *only* the following front matter, starting with '---' and ending with '---'. Do not include anything else.

Format:
---
title: '${POST_TITLE}'
date: ${CURRENT_DATE}
permalink: /blog/${CURRENT_YEAR}/${CURRENT_MONTH}/short-title/
tags:
  - example-tag-1
  - example-tag-2
---
EOM

# LLM call
echo "Generating front matter using LLM..."
FRONT_MATTER=$($LLM_COMMAND -m "$MODEL_NAME" "$LLM_PROMPT")

# remove thinking tokens and then empty newlines
FRONT_MATTER=$(echo "$FRONT_MATTER" | perl -0777 -pe 's|<think>.*?</think>||gs; s/^\s*\n+//')

if [ -z "$FRONT_MATTER" ]; then
    echo "Error: Failed to get response from LLM."
    exit 1
fi

# extract short-title
SHORT_TITLE=$(echo "$FRONT_MATTER" | grep 'permalink:' | sed 's#.*/\([^/]*\)/#\1#')

if [ -z "$SHORT_TITLE" ]; then
    echo "Error: Could not extract short-title from LLM output:"
    echo "--- LLM Output ---"
    echo "$FRONT_MATTER"
    echo "------------------"
    exit 1
fi

# attempt to write to file but throw error if file exists
FILENAME="${POSTS_DIR}/${CURRENT_DATE}-${SHORT_TITLE}.md"

mkdir -p "$POSTS_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Could not create directory '$POSTS_DIR'."
    exit 1
fi

if [ -f "$FILENAME" ]; then
    echo "Error: File already exists: '$FILENAME'" >&2 
    exit 1 
fi

echo "$FRONT_MATTER" > "$FILENAME"

# report out
if [ $? -eq 0 ]; then
    echo "Successfully created blog post file:"
    echo "$FILENAME"
    echo "With content:"
    cat "$FILENAME"
else
    echo "Error: Failed to write to file '$FILENAME'."
    exit 1
fi

exit 0