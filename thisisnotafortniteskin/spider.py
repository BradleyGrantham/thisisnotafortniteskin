"""Crawl https://fortniteskins.net/ for images of current fortnite skins."""
import os
import shutil

import bs4
import click
import requests
from tqdm import tqdm

import thisisnotafortniteskin.constants

LOGGER = thisisnotafortniteskin.constants.logger


def soupify(url: str):
    r = requests.get(url)
    if r.status_code == 200:
        soup = bs4.BeautifulSoup(r.content, "html.parser")
        return soup
    else:
        return r.status_code


def pages_of_skins(starting_page=1):
    url = "https://fortniteskins.net/outfits/page/{}"
    page = starting_page
    page_errors = []
    while len(page_errors) < 2:
        status_code = requests.get(url.format(page)).status_code
        if status_code == 200:
            yield url.format(page)
        else:
            LOGGER.warning(
                "{} returned status code {}".format(url.format(page), status_code)
            )
            page_errors.append(status_code)
        page += 1


def get_skin_links(page_of_skins: str):
    soup = soupify(page_of_skins)
    assert isinstance(soup, bs4.BeautifulSoup), f"Error retrieving {page_of_skins}"

    image_entries = soup.find_all("div", {"class": "e-skin__block"})
    for image_entry in image_entries:
        yield image_entry.find("a").get("href", None)


def get_skin_png(single_skin_page: str, save_directory: str):
    skin_name = single_skin_page.split("/")[-2]
    soup = soupify(single_skin_page)
    image_url = (
        soup.find("div", {"class": "skin__main-image"}).find("img").get("src", None)
    )
    if image_url is not None:
        save_image_from_url(image_url, os.path.join(save_directory, f"{skin_name}.png"))


def save_image_from_url(url: str, output_file: str):
    with requests.session() as sess:
        r = sess.get(url, stream=True)
        with open(output_file, "wb") as f:
            shutil.copyfileobj(r.raw, f)


@click.command()
@click.argument("save-directory")
def crawl(save_directory: str):
    LOGGER.info("Beginning to crawl outfit pages")
    if not os.path.exists(save_directory):
        LOGGER.info("Creating save directory")
        os.makedirs(save_directory, exist_ok=True)

    skin_page_urls = [
        skin_link
        for url in pages_of_skins()
        for skin_link in get_skin_links(url)
    ]
    LOGGER.info("Obtained {} total outfits".format(len(skin_page_urls)))

    LOGGER.info("Scraping individual outfit pages and saving images")
    for skin_link in tqdm(skin_page_urls):
        get_skin_png(skin_link, save_directory=save_directory)


if __name__ == "__main__":
    crawl()
