import os
import importlib
import asyncio

import numpy as np
import faiss

os.environ.setdefault('OPENAI_API_KEY', 'test')
os.environ.setdefault('SERPAPI_API_KEY', 'test')

import investment_agents as ia

# reload to ensure environment vars are read
ia = importlib.reload(ia)

def setup_simple_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ia, '_embed', lambda text: np.ones(ia.EMBED_DIM, dtype='float32'))
    from vector_store import VectorStore
    store = VectorStore(str(tmp_path/'index.bin'), str(tmp_path/'meta.json'), ia.EMBED_DIM)
    store.load()
    monkeypatch.setattr(ia, 'vstore', store)
    monkeypatch.setattr(ia, 'INDEX_PATH', str(tmp_path/'index.bin'))
    monkeypatch.setattr(ia, 'META_PATH', str(tmp_path/'meta.json'))
    monkeypatch.setattr(ia, 'REPORT_DIR', str(tmp_path))


def fake_run_sync(agent, text):
    class Res:
        def __init__(self, out):
            self.final_output = out
    if agent.name == 'ReportAgent':
        return Res('# Summary\ns\n\n# Keywords\n- k\n\n# Metrics\n- m: 1\n\n# Decision\nYES\n\n# Rationale\nall good')
    if agent.name == 'SupervisorAgent':
        return Res('YES\nall good')
    return Res(f'result from {agent.name}')


async def fake_run_async(agent, text):
    await asyncio.sleep(0)
    return fake_run_sync(agent, text)


def test_evaluate_creates_markdown(tmp_path, monkeypatch):
    setup_simple_env(tmp_path, monkeypatch)
    monkeypatch.setattr(ia, '_extract_pdf', lambda pdf: 'text')
    monkeypatch.setattr(ia.Runner, 'run_sync', staticmethod(fake_run_sync))
    monkeypatch.setattr(ia.Runner, 'run', staticmethod(fake_run_async))
    result = asyncio.run(ia.evaluate('dummy.pdf', 'proj1'))
    assert result.summary == 's'
    assert result.decision == 'YES'
    assert os.path.exists(result.markdown)
    with open(result.markdown, 'r', encoding='utf-8') as fh:
        assert '# Summary' in fh.read()


def test_vector_memory_add_query_list(tmp_path, monkeypatch):
    setup_simple_env(tmp_path, monkeypatch)
    monkeypatch.setattr(faiss, 'write_index', lambda index, path: None)
    assert ia.vector_memory_impl('add', project='p', summary='sum', keywords=['kw'], rationale='rat') == 'stored'
    q = ia.vector_memory_impl('query', summary='sum')
    assert q and q[0]['summary'] == 'sum'
    assert ia.vector_memory_impl('list') == ia.vstore.list()
