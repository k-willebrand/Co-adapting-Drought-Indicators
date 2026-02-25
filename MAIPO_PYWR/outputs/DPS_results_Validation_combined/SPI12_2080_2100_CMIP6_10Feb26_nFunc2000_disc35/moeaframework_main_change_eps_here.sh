#!/bin/bash

# The only thing you need to change in this file
epsilon="0.01,0.01,50000000"

# Check if Java is installed and callable
# Please download and install the Java Development Kit (JDK) from:
# https://www.oracle.com/java/technologies/downloads/#jdk23-windows

# Check the expected JAR file
jarFile=""
jarURL="https://github.com/MOEAFramework/MOEAFramework/releases/download/v4.5/MOEAFramework-4.5-Demo.jar"
jarName="MOEAFramework-4.5-Demo.jar"

# Check if the jar file exists
for file in *Demo.jar; do
  jarFile="$file"
done

if [ -z "$jarFile" ]; then
  echo
  echo "[ERROR] MOEAFramework Demo JAR file not found in the current directory."
  echo "The required file can be downloaded from:"
  echo "$jarURL"
  echo
  
  echo "Downloading $jarName..."
  curl -L "$jarURL" -o "$jarName"

  if [ -f "$jarName" ]; then
    echo "Download complete."
    jarFile="$jarName"
  else
    echo "[ERROR] Failed to download the file."
    exit 1
  fi
fi

# Java and JAR file found
echo "Java is installed and MOEAFramework JAR file found: $jarFile"
echo "Proceeding with execution..."

# Main code
echo "Running step 1: Merging result files..."
# Count the number of elements in the epsilon array to set dimension
dimension=0
IFS=',' read -ra epsilon_array <<< "$epsilon"
for _ in "${epsilon_array[@]}"; do
  ((dimension++))
done

# Loop over all .runtime files in the current directory
for input_file in *.runtime; do
  output_file="${input_file%.runtime}.set"
  echo "Processing $input_file"
  
  java -cp "$jarFile" org.moeaframework.analysis.tools.ResultFileMerger \
    --dimension "$dimension" \
    --output "$output_file" \
    --epsilon "$epsilon" \
    "$input_file"
done

echo "Running step 2: Merging reference set..."
# Loop over all .set files in the current directory
for set_file in *.set; do
  echo "Processing $set_file"

  java -cp "$jarFile" org.moeaframework.analysis.tools.ReferenceSetMerger \
    --output borg.ref \
    --epsilon "$epsilon" \
    "$set_file"
done

echo "Running step 3: Evaluating result files..."
# Loop over all .runtime files in the current directory
for input_file in *.runtime; do
  output_file="${input_file%.runtime}.metrics"
  echo "Evaluating $input_file"
  
  java -cp "$jarFile" org.moeaframework.analysis.tools.ResultFileEvaluator \
    --dimension "$dimension" \
    --epsilon "$epsilon" \
    --input "$input_file" \
    --reference borg.ref \
    --output "$output_file"
done

echo "All tasks completed successfully."