# IR
from bs4 import BeautifulSoup
from urllib import parse, request


def get_local_domain(site):
    if len(site) < 4:
        return site
    if site[:4] == "www.":
        return site[4:]
    return site


def get_links(root, html):
    root_domain = get_local_domain(parse.urlparse(root)[1])
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        if link.get('href'):
            curr_domain = get_local_domain(parse.urlparse(link.get('href'))[1])
            if root_domain and curr_domain and curr_domain != root_domain:
                text = ""
                if link.string:
                    text = link.string
                yield (parse.urljoin(root, link.get('href')), text)


def main():
    site = 'http://www.cs.jhu.edu/~yarowsky'
    r = request.urlopen(site)
    for l in get_links(site, r.read()):
        print(l)


if __name__ == '__main__':
    main()
