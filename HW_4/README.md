## IR HW4
### Anant Bhargava
### Prakhar Kaushik


## P1


## P2


Some webpages are accessed by using ``http://`` and some by ``https://``, also links might sometimes have ``www.`` . To ensure that multiple visits to the same site are not allowed due to using ``http://`` once and  ``https://``, similarly for ``www.``, the protocol is parsed out from the url and if ``www.`` is removed if present and standard ``http://`` is used.

All the html pages which are visited have the links printed to ``pages_visited.txt``, and information extracted from them is in ``extracted_info.txt``, pdfs which were visited are in ``pdfs_visited.txt``. 

#TODO: MAKE IT ARGPARSE!!
To adjust level of recursion please set ``terminate_level`` parameter in function call to crawl in the main function (starting address goes in the ``root`` parameter) 
 
