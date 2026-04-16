import requests
import json
import pandas as pd

# ============================================================
# FUNCTION: submit batch to ZTF
# ============================================================
def submit_post(ra_list, dec_list):
    payload = {
        'ra': json.dumps(ra_list),
        'dec': json.dumps(dec_list),
        'jdstart': json.dumps(2458216.1234),
        'jdend': json.dumps(2458450.0253),
        'email': 'gmd5786@psu.edu',
        'userpass': 'zfps095'
    }

    url = 'https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit'

    try:
        r = requests.post(url, auth=('ztffps', 'dontgocrazy!'), data=payload)
        print("Submitted batch of", len(ra_list), "objects | Status_code =", r.status_code)
    except Exception as e:
        print("Request failed:", e)


# ============================================================
# READ CSV (this fixes your original error)
# ============================================================
df = pd.read_csv("confirmedclagn.csv") 

# Ensure correct column names
if "meanra" not in df.columns or "meandec" not in df.columns:
    raise ValueError("CSV must contain 'meanra' and 'meandec' columns")

# Round values (like you were doing)
ralist_all = df["meanra"].astype(float).round(7).tolist()
declist_all = df["meandec"].astype(float).round(7).tolist()

print("Number of (ra,dec) pairs =", len(ralist_all))


# ============================================================
# BATCH SUBMISSION (max 1500 per request)
# ============================================================
batch_size = 1500

for i in range(0, len(ralist_all), batch_size):
    ra_batch = ralist_all[i:i+batch_size]
    dec_batch = declist_all[i:i+batch_size]

    submit_post(ra_batch, dec_batch)

print("All submissions complete.")



#!/usr/bin/env python3
import re
import requests
# Script name: confirmed_CLAGN_phot.py
settings = {'email': 'gmd5786@psu.edu', 'userpass': 'zfps095',
            'option': 'All recent jobs', 'action': 'Query Database'}
r = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/' +\
                 'getBatchForcedPhotometryRequests.cgi',
                 auth=('ztffps', 'dontgocrazy!'), params=settings)
#print(r.text)
if r.status_code == 200:
    print("Script executed normally and queried the ZTF Batch " +\
    "Forced Photometry database.\n")
    wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! -O '
    wget_url = 'https://ztfweb.ipac.caltech.edu'
    wget_suffix = '"'
    lightcurves = re.findall(r'/ztf/ops.+?lc.txt\b',r.text)
    if lightcurves is not None:
        for lc in lightcurves:
            p = re.match(r'.+/(.+)', lc)
            fileonly = p.group(1)
            print(wget_prefix + " " + fileonly + " \"" + wget_url + lc +\
            wget_suffix)
else:
    print("Status_code=",r.status_code,"; Jobs either queued or" +\
    "abnormal execution.")
