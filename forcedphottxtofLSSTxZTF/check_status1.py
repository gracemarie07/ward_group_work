#!/usr/bin/env python3
import re
import requests
# Script name: check_status1.py
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
