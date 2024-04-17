import re

FILE_NAME = 'entrada.txt'

def main():
    with open(FILE_NAME, 'r') as file:
        info  = file.read()
        
        words = re.findall(r"(?:\w|')+", info)
        
        for word in words:
            print(word.lower())
        
            
        
if __name__ == '__main__':
    main()