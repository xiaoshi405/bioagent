"""Run the existing fixture-based scientific-object identity demonstration."""
from pathlib import Path
import asyncio,sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper_identity_probe import main

if __name__=='__main__':asyncio.run(main())
