import psutil
import csv
import os
import requests
import time
import json
import re
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

def handle_rate_limit(response):
    if response.headers.get('x-ratelimit-remaining') == '1':
        time.sleep(60)

def extract_repo_metadata(item):
    return {
        'full_name': item['full_name'],
        'stars': item['stargazers_count'],
        'forks': item['forks_count'],
        'topics': item['topics'],
        'language': item['language'],
        'created_at': item['created_at'],
        'pushed_at': item['pushed_at'],
        'updated_at': item['updated_at'],
        'watchers_count': item['watchers_count'],
        'open_issues_count': item['open_issues_count'],
        'subscribers_count': item['subscribers_count'],
        'default_branch': item['default_branch']
    }

def fetch_page_of_repos(url, auth_params, results, n_repos, search_params):
    response = requests.get(url, auth=auth_params)
    handle_rate_limit(response)
    if n_repos is not None:
        page_num = int(url.split('&page=')[1])
        remaining = n_repos - (search_params['per_page'] * page_num)
        if remaining <= 0:
            limit = search_params['per_page'] + remaining
            for item in response.json()['items'][:limit]:
                results.append(extract_repo_metadata(item))
            return True
    for item in response.json()['items']:
        results.append(item)
    return False


def get_repos(search_params: dict, auth_params, n_repos=None):
    """
    Returns a list of repos based on given search criteria. Uses keyword search.
    Args:
        search_params:  Dictionary of paremeters for the request to the GitHub REST API.
        auth_params:    Tuple (username, token). Used for authentication to GitHub.
        n_repos:        Maximum number of repositories to be returned. None means that all repositories that are found  will be returned.
    Returns:    List of dictionaries containing the metadata for the repositories found.
    """
    url = "https://api.github.com/search/repositories"
    response = requests.get(url, params=search_params, auth=auth_params)
    handle_rate_limit(response)
    current_page = response.url
    n_results = response.json()['total_count']
    try:
        last_page = response.links['last']['url']
    except KeyError:
        last_page = current_page

    results = []
    while True:
        print(current_page)
        
        stop_fetching = fetch_page_of_repos(current_page, auth_params, results, n_repos, search_params)
        if stop_fetching:
            break

        if current_page != last_page:
            current_page = response.links['next']['url']
        else:
            break
    return (results, n_results)

def get_list_from_file(file):
    with open(file) as f:
        rv = f.readlines()
    return [i.replace("\n", "") for i in rv]

def handle_api_response(response_object, url):
    if response_object.status_code == 200:
        return response_object
    elif response_object.status_code == 429:
        retry_after = int(response_object.raw.headers._container['retry-after'][1]) + 10
        print(f"429 (too many requests): {url} retrying after {retry_after} s")
        time.sleep(retry_after)
    elif response_object.status_code == 403:
        if "Repository access blocked" in response_object.text:
            return None
        reset_epoch = response_object.raw.headers._container['x-ratelimit-reset'][1]
        ratelimit_remaining = int(response_object.raw.headers._container['x-ratelimit-remaining'][1])
        current_epoch = int(time.time())
        to_wait = int(reset_epoch) - current_epoch + 10
        if ratelimit_remaining == 0:
            print(f"403 (forbidden): {url} retrying after {to_wait} s")
            time.sleep(to_wait)
    elif response_object.status_code in (404, 451, 409):
        return None
    else:
        with open("failed_repos.txt", 'a+') as f:
            f.write(f"{response_object.status_code},{url}\n")
        return None
    return None

def make_api_request(url, auth_params, req_params=None):
    while True:
        try:
            if req_params:
                response_object = requests.get(url, auth=auth_params, params=req_params)
            else:
                response_object = requests.get(url, auth=auth_params)
        except requests.exceptions.ConnectionError:
            print("Connection dropped. Sleeping for 5 minutes before retrying.")
            time.sleep(60 * 5)
            continue

        response_object = handle_api_response(response_object, url)
        if response_object:
            return response_object
        if response_object is None:
            return None

def safe_api_query(url, auth_params, req_params=None):
    """
    Queries a GitHub API URL and handles the reponse status codes:
    Args:
        url:                String. GitHub API URL.
        auth_params:        Tuple (username, token). Used for authentication to GitHub.
        req_params:         Dict. Additional request parameters to make the request to the GitHub API with,
                            e.g.  {'per_page': 100}.
    Returns:                Response object if successful, None if repository cannot be found or has been blocked.
    """
    return make_api_request(url, auth_params, req_params)

def query_repo(repo, auth_params):
    """
    Queries the GitHub API for a repository and returns the information about it as a dict, otherwise None
    if the repository is blocked or cannot be found.
    Args:
        repo:               String. Library name on the format owner_username/repository_name, e.g. tensorflow/tensorflow.
        auth_params:        Tuple (username, token). Used for authentication to GitHub.
    Returns:                Dict if successful, None if repository cannot be found or has been blocked.
    """
    url = f"https://api.github.com/repos/{repo}"
    response_object = safe_api_query(url, auth_params)
    if response_object is not None:
        return json.loads(response_object.content)
    else:
        return None

def retrieve_n_commits(repo, auth_params):
    """
    Determine the number of commits for a repository using the GitHub REST API.
    Args:
        repo_url:       String. Library name on the format owner_username/repository_name, e.g. tensorflow/tensorflow.
        auth_params:    Tuple (username, token). Used for authentication to GitHub.
    Returns:            Integer if successful, None if repository cannot be found or has been blocked.
    """
    url = f"https://api.github.com/repos/{repo}/commits?per_page=1"
    response_object = safe_api_query(url, auth_params)
    if response_object is not None:
        try:
            n_commits = re.findall(r"\d+", response_object.links['last']['url'])[0]
            return int(n_commits)
        except (KeyError, IndexError, ValueError):
            return None
    else:
        return None

def list_to_str(lst):
    return "|".join(lst)

tracker.stop()
