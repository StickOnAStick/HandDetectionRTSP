import sys
import os
from app import CVStreamApp

def main() -> int:

    app = CVStreamApp()
    app.run()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())