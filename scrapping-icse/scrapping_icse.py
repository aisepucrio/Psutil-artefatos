from bs4 import BeautifulSoup
import requests
import csv

url = "https://conf.researchr.org/track/icse-2025/icse-2025-research-track?#event-overview"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', class_='table table-condensed')

data = []

if table:
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue

        for td in tds:
            has_artifact = td.find('img', alt="Artifact-Available") is not None
            titles = []
            authors = []

            for link in td.find_all('a', href=True):
                if link['href'].startswith('#'):
                    text = link.get_text(strip=True)
                    if text:
                        titles.append(text)

            for performer in td.find_all(class_='performers'):
                text = performer.get_text(strip=True)
                if text:
                    authors.append(text)

            for title in titles:
                data.append({
                    'Title': title,
                    'Authors': ', '.join(authors),
                    'HasArtifact': has_artifact
                })

    with open('icse2025_artifacts.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Authors', 'HasArtifact']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print("file 'icse2025_artifacts.csv' save successfully.")

