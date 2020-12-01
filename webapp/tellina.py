# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

from bs4 import BeautifulSoup
import requests
import urllib

ip = requests.get('https://api.ipify.org').text


def scrape(query):
	parm = urllib.parse.urlencode({'request_str':query})
	url = "http://kirin.cs.washington.edu:8000/translate?"+parm
	page = requests.get(url, cookies={'ip_address':ip})
	soup = BeautifulSoup(page.content)
	code = soup.find('code')
	cmd = " ".join([x.text for x in code.find_all('span')])
	return cmd
