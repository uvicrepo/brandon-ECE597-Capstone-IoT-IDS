# scripts/generate_dataset.py

Generates sampled packet and flow datasets for Phase 1 and Phase 3. Outputs two files to `data/sampled/`:
- `packet_sample.parquet` - used in Phase 2 (unsupervised detection)
- `flow_sample.parquet` - used in Phase 3 (supervised classification)

Both samples contain 200,000 benign rows and 4,000–6,200 attack rows, randomly distributed across the five attack types.

---

**From the CLI** (run from repo root):
```bash
python scripts/generate_dataset.py
```
To change the seed, edit `SEED` at the bottom of the script. The script will download the dataset automatically if it hasn't been downloaded yet.

**In a notebook:**
```python
import sys; sys.path.insert(0, "../scripts")
from generate_dataset import sample_dataset

packet_sample = sample_dataset("packet", seed=42)
flow_sample = sample_dataset("flow", seed=42)
```

---

**Dataset filename quirks** - the original CIC IoT-DIAD 2024 files have a few inconsistent names that are preserved as-is so everyone on the team gets the same files when they download the dataset:

- `DDoS-HTTP_Flood-` has a trailing dash in the filename. This is an artifact of how the dataset was originally exported and not a typo.
- Brute force traffic is stored in a file called `DictionaryBruteForce`, not `BruteForce`. A dictionary attack is a specific type of brute force, but the file won't be where you expect it.
- DoS and benign traffic are each split across two and four files respectively, because they were captured in multiple sessions. The script combines them before sampling.
- Flow files have `.pcap_Flow` in their name (e.g. `DNS_Spoofing.pcap_Flow.parquet`) reflecting the tool used to extract flow features from the raw packet captures.

These mappings are defined in `dataImport.toml` and referenced in `ATTACK_KEYS` near the top of the script.