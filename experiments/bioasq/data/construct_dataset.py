import glob
import json
import tqdm
import argparse
import pandas as pd

from collections import defaultdict 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to allMesh.json')

    args = parser.parse_args()

    file_pmids_list = defaultdict(dict)
    file_pmids_lookup = defaultdict(dict)
    for fname in glob.glob('subsets-v-20000/*.csv'):
        print('Processing %s...' % fname)
        df = pd.read_csv(fname)
        pmids = list(pmid for pmid in df.pmid)
        file_pmids_list[fname] = pmids
        file_pmids_lookup[fname] = set(pmids)

    file_pmids_df = defaultdict(dict)
    with open(args.data, 'rb') as f:
        next(f)
        for i, line in enumerate(tqdm.tqdm(f)):
            line = line.decode('latin-1').rstrip()
            # Last line
            if line.endswith('}]}'):
                # Ignore closing of "articles"
                line = line[:-2]
            else:
                # Ignore comma
                line = line[:-1]
            entry = json.loads(line)
            for key in file_pmids_lookup:
                pmid = entry['pmid']
                if int(pmid) in file_pmids_lookup[key]:
                    file_pmids_df[key][int(pmid)] = entry
    for key, entries in file_pmids_df.items():
        assert(len(entries) == len(file_pmids_list[key]))
        with open(key.replace('.csv', '.json'), 'a') as f:
            for pmid in file_pmids_list[key]:
                if pmid in entries:
                    f.write('%s\n' % json.dumps(entries[pmid]))
