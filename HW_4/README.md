## IR HW4
### Anant Bhargava
### Prakhar Kaushik


## P1
Implemented as desired to only get the non local links and print out the standard output. To run use:
``python3 p1.py -site="SITEADDRESS"`` if no arguments given defaults to: ``http://www.cs.jhu.edu/~yarowsky``

## P2


Some webpages are accessed by using ``http://`` and some by ``https://``, also links might sometimes have ``www.`` . To ensure that multiple visits to the same site are not allowed due to using ``http://`` once and  ``https://``, similarly for ``www.``, the protocol is parsed out from the url and if ``www.`` is removed if present and standard ``http://`` is used.

All the html pages which are visited have the links printed to ``pages_visited.txt``, and information extracted from them is in ``extracted_info.txt``, pdfs which were visited are in ``pdfs_visited.txt``. 

Instructions to run the crawler: 
``python3 p2.py `` specify ``root="DESIRED ROOT"`` which defaults to ``http://www.cs.jhu.edu/~yarowsky/cs466.html``, and ``terminate-level=max recursion`` which by default is 3, meaning BFS with depth of 2.  Example: ``python3 p2.py root="http://www.cs.jhu.edu/~yarowsky/cs466.html" terminate-level=10000``

To calculate relevancy text of current HTML is taken, and then SequenceMatcher from difflib is used with the target html page and the similarity is used as factor to update the priority. 

 
