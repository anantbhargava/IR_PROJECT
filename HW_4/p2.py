from queue import PriorityQueue
import re
from bs4 import BeautifulSoup
from urllib import parse, request
from difflib import SequenceMatcher
import logging
from p1 import get_local_domain
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

results = []

extract_fid = open('extracted_info.txt', 'w')
links_visited = open('pages_visited.txt', 'w')
pdf_visited = open('pdfs_visited.txt', 'w')


# Extracts the phone number, email address
def extract(address, html):
    global extract_fid
    # This gets the phone numbers
    for match in re.findall(r'\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        logging.debug('Found phone: {}, address: {}'.format(match, address))
        extract_fid.writelines('; '.join([address, 'PHONE', match]) + '\n')

    # This gets the contact info as mentioned
    for match in re.findall(r'[a-zA-z.]*,\s* .[a-zA-z.]*\s*\d{5}', str(html)):
        logging.debug('Found contact address: {}, address: {}'.format(match, address))
        extract_fid.writelines('; '.join([address, 'CONTACT', match]) + '\n')

    # This gets the mail id
    for match in re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", str(html)):
        match = match.strip()
        logging.debug('Found contact email: {}, address: {}'.format(match, address))
        extract_fid.writelines('; '.join([address, 'EMAIL', match]) + '\n')
    extract_fid.flush()


# To get links
def get_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        if link.get('href'):
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)


def process_address(address):
    parsed_url = parse.urlparse(address)
    domain = get_local_domain(parsed_url[1])
    max_len = address.find(domain) + len(domain)
    return "http://" + domain + address[max_len:]


# root is the portion to start from

def crawl(root, terminate_level=2):
    visited = set()
    global links_visited

    def shouldvisit(address):
        if (address not in visited) and get_local_domain(parse.urlparse(address)[1]) == "cs.jhu.edu":
            return True
        return False

    def wanted(req):
        if 'text/html' in req.headers['Content-Type']:
            return 1
        elif 'pdf' in req.headers['Content-Type']:
            return 2
        return -1

    def relevance(comp_str, curr_address):
        try:
            n_r = request.urlopen(curr_address)
            if (n_r.status == 200) and (wanted(n_r) == 1):
                content_2 = BeautifulSoup(n_r.read(), 'html.parser').text
                return SequenceMatcher(None, content_2, comp_str).ratio()
        except:
            return 0.001
        return 0

    queue = PriorityQueue()
    queue.put((0, process_address(root)))

    # Loop till not complete
    while not queue.empty():
        priority, address = queue.get()

        if priority > terminate_level:
            break

        try:
            r = request.urlopen(address)
            if r.status == 200:
                page_type = wanted(r)
                if page_type == 2:
                    pdf_visited.writelines(address + '\n')
                elif page_type == 1:
                    links_visited.writelines(address + '\n')
                    logging.info('Popped off the queue: {}, Priority: {}'.format(address, priority))
                    html = r.read()
                    extract(address, html)
                    content_1 = BeautifulSoup(html, 'html.parser').text

                    for link, title in get_links(address, html):
                        link = process_address(link)
                        if shouldvisit(link):
                            visited.add(link)
                            if priority + 1 < terminate_level:
                                new_priority = priority + relevance(content_1, link) + 1
                                queue.put((new_priority, link))
                                logging.debug('Pushed on queue URL: {} Priority: {}'.format(link, new_priority))
            links_visited.flush()
            pdf_visited.flush()
        except Exception as e:
            print(e, address)


def main():
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level='INFO')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Robot Crawler')
    parser.add_argument("-root", type=str, default="http://www.cs.jhu.edu/~yarowsky/cs466.html", help='Root Site')
    parser.add_argument("-terminate_level", type=int, default=4, help='Terminate after x recursions')
    args = parser.parse_args()
    crawl(root=args.root, terminate_level=args.terminate_level)
    extract_fid.close()
    links_visited.close()


if __name__ == '__main__':
    main()
