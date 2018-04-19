# IR
from bs4 import BeautifulSoup
from urllib import parse, request
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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
                    text = link.string.strip()
                yield (parse.urljoin(root, link.get('href')), text)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='P1')
    parser.add_argument("-site", type=str, default="http://www.cs.jhu.edu/~yarowsky", help='Site')
    args = parser.parse_args()
    r = request.urlopen(args.site)
    for l in get_links(args.site, r.read()):
        print(l)


if __name__ == '__main__':
    main()
