"""Functions to process OpenAI prompts with cache (adapted from Tidybot source)."""
import os
import pickle
import tempfile
import time
from pathlib import Path

import openai
from openai import OpenAI

API_KEY = os.getenv('OPENAI_API_KEY')

class Completion:
    def __init__(self, cache_folder='.openai'):
        # Note: Cache is not threadsafe, do not call create() in multiple places at the same time
        #self.cache_path = Path(cache_path)
        self.cache_folder = Path(__file__).resolve().parent / cache_folder
        self.cache_path = self.cache_folder / "openai_cache"
        self.cache_mtime = None
        if self.cache_path.exists():
            self._load_cache_file()
        else:
            self.cache = {}
            self.cache_folder.mkdir(exist_ok=True)
            self._save_cache_file()

        # Handle rate limits
        self.last_request_time = 0
        self.client = OpenAI(api_key=API_KEY)

    def create(self, prompt, model='gpt-4-turbo-preview'):
        if self._check_cache_file_modified():
            self._load_cache_file()

        if model not in self.cache:
            self.cache[model] = {}
        if prompt not in self.cache[model]:
            requests_per_min = 20 if model.startswith('code-') else 3000
            while time.time() - self.last_request_time < 60 / requests_per_min:
                time.sleep(0.001)
            success = False
            while not success:
                try:
                    self.last_request_time = time.time()
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=1024,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    success = True
                except openai.RateLimitError:
                    time.sleep(60 / requests_per_min)
            if self._check_cache_file_modified():
                self._load_cache_file()
            self.cache[model][prompt] = response
            self._save_cache_file()

        return self.cache[model][prompt]

    def _load_cache_file(self):
        self.cache_mtime = os.path.getmtime(self.cache_path)
        with open(self.cache_path, 'rb') as f:
            self.cache = pickle.load(f)

    def _save_cache_file(self):
        #start_time = time.time()
        tmp_folder = self.cache_folder / "tmp"
        tmp_folder.mkdir(exist_ok=True)
        tmp_cache_path = tmp_folder / self.cache_path.name
        with open(tmp_cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
        os.replace(tmp_cache_path, self.cache_path)
        self.cache_mtime = os.path.getmtime(self.cache_path)
        #print(f'Saving cache file ({len(self.cache)} entries) took {1000 * (time.time() - start_time):.1f} ms')

    def _check_cache_file_modified(self):
        if os.path.getmtime(self.cache_path) > self.cache_mtime:
            print('Warning: Cache file unexpectedly modified, race condition possible')
            return True
        return False

    def dump_cache_to_txt(self, dump_path='openai-cache.txt'):
        with open(dump_path, 'w', encoding='utf8') as f:
            for model in self.cache.keys():
                f.write(model + '\n')
                f.write(80 * '=' + '\n')
                for prompt in self.cache[model].keys():
                    f.write(prompt + '\n')
                    f.write(80 * '-' + '\n')
        print(f'Dumped cache to {dump_path}')

if __name__ == '__main__':
    # Race condition test
    # from threading import Thread
    # def f(i):
    #     completion = Completion(cache_path='tmp-cache.pkl')
    #     for j in range(3):
    #         print(f'Worker {i} completion {j}')
    #         completion.create(f'Worker {i} says {j}', model='text-ada-001')
    # threads = [Thread(target=f, args=(i,)) for i in range(2)]
    # [t.start() for t in threads]
    # [t.join() for t in threads]

    completion = Completion()
    completion.dump_cache_to_txt()
