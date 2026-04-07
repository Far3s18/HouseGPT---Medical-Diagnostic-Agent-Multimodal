import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_files = [
    '.env',
    '.pre-commit-config.yaml',
    '.gitignore',
    'docs/readme.md',
    'images/',
    'notebooks/trial.ipynb',
    'src/house_gpt/__init__.py',
    'src/house_gpt/core/__init__.py',
    'src/house_gpt/core/exceptions.py',
    'src/house_gpt/core/logger.py',
    'src/house_gpt/core/settings.py',
    'src/house_gpt/agent/prompts/__init__.py',
    'src/house_gpt/agent/prompts/character.py',
    'src/house_gpt/agent/prompts/router.py',
    'src/house_gpt/agent/prompts/memory.py',
    'src/house_gpt/agent/chains/__init__.py',
    'src/house_gpt/agent/chains/chains.py',
    'src/house_gpt/agent/graph/__init__.py',
    'src/house_gpt/agent/graph/graph.py',
    'src/house_gpt/agent/graph/nodes.py',
    'src/house_gpt/agent/graph/edges.py',
    'src/house_gpt/agent/helpers/__init__.py',
    'src/house_gpt/agent/helpers/formatter.py',
    'src/house_gpt/agent/helpers/model_factory.py',
    'src/house_gpt/schedules/__init__.py',
    'src/house_gpt/schedules/schedules.py',
    'src/house_gpt/schedules/context_generation.py',
    'src/house_gpt/states/__init__.py',
    'src/house_gpt/states/house.py',
    'src/house_gpt/states/memory.py',
    'src/house_gpt/states/response.py',
    'src/house_gpt/memory/ltm/__init__.py',
    'src/house_gpt/memory/ltm/memory_manager.py',
    'src/house_gpt/memory/ltm/vector_store.py',
    'src/house_gpt/memory/models/__init__.py',
    'src/house_gpt/memory/models/memory_model.py',
    'src/house_gpt/multimodal/__init__.py',
    'src/house_gpt/multimodal/image/__init__.py',
    'src/house_gpt/multimodal/image/image_to_text.py',
    'src/house_gpt/multimodal/image/text_to_image.py',
    'src/house_gpt/multimodal/speech/__init__.py',
    'src/house_gpt/multimodal/speech/speech_to_text.py',
    'src/house_gpt/multimodal/speech/text_to_speech.py',
]

for filepath in list_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Ceating directory: {filedir} for the file: {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file: {filepath}')

    else:
        logging.info(f'{filename} is aleready exists')