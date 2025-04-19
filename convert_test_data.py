def convert_test_data():
    input_file = "testing_annotated_data"
    output_file = "testing_data.csv"
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Write header to output CSV
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("Sent,Word,Tag\n")
        
        # Read input file and convert
        current_sent = 0
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    if not line:  # Empty line indicates new sentence
                        current_sent += 1
                    continue
                
                # Split line into word and tag
                parts = line.split()
                if len(parts) >= 2:
                    # Join all parts except the last one as the word
                    word = " ".join(parts[:-1])
                    tag = parts[-1]
                    
                    # Handle special characters in word
                    if any(c in word for c in ',"\''):
                        # Replace problematic characters
                        word = word.replace('"', '')
                        word = word.replace(',', '')
                        word = word.replace('\'', '')
                    
                    # Write to CSV format, ensuring word is properly escaped
                    out_f.write(f'sent: {current_sent},{word},{tag}\n')
    
    print("Conversion completed!")
    print(f"Created {output_file} with proper CSV format for testing")

if __name__ == "__main__":
    convert_test_data() 