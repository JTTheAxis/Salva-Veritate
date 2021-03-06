#-----------------------------------------------------------------------------
# Name:        newsitem.py
# Purpose:     To store the NewsItem class.                   
# Author:      Jason Tian
# Created:     05-Jun-2020
# Updated:     05-Jun-2020 
#-----------------------------------------------------------------------------

class NewsItem():
    '''
    Any instance of news (report, article, etc) contained in a single file. 
    
    Attributes
    ----------
    keywordCount : int
    	The number of words designated as keywords which are contained in many fake news articles.
    author : str
    	Name of the individual who wrote this news file.
    org : str
    	Name of the organization for which this file was written.
    topic : str
    	The main subject under which this file could be classified.
    title : str
    	The title of the work.
    date : str
    	The date when the work was officially published.
    content : str
    	The entirety of the news work's text.
    valid : bool
    	Whether or not the work is "true".
        
    Methods
    -------
    countKeywords(keywordList : list) -> int
    	Counts the number of words considered indicative of false news works within this particular work.
    printAuthor() -> None
    	Prints the author of the work.
    printDate() -> None
    	Prints the date in which the work was published
    read() -> None
    	Prints the entirety of the news work's test for the user to read.
    dubRatio() -> float
    	Returns the ratio of words considered indicative of a false work to total words.
    
    '''
    
    def __init__(self, title, content, valid, keywordCount=0, author="?", org="?", topic="?", date="?"):
        '''
        Constructor to build a NewsItem object.
        
        Parameters
        ----------
        title : str
        	Title of the work.
        content : str
    		The entirety of the news work's text.
    	valid : bool
        	Whether or not the work is "true".
        keywordCount : int, optional
        	Number of common words in fake news found in this work.
                Set to 0 if not calculated or unknown beforehand.
        author : str, optional
        	Author of the work.
                Placeholder is "?" if unknown.
        org : str, optional
        	Organization for which the work was published.
                Placeholder is "?" if unknown.
        topic : str, optional
        	Main subject of the work.
                Placeholder is "?" if unknown.
        date : str, optional
        	The work's date of publication.
                Placeholder is "?" if unknown.
                
        '''
        
        self.title=title
        self.keywordCount=keywordCount
        self.valid=valid
        self.author=author
        self.org=org
        self.topic=topic
        self.date=date
        self.content=content
    
    def countKeywords(self, keywordList : list) -> int:
        '''
        Counts the number of identified keywords in the text of the work.
        
        Parameters
        ----------
        keywordList : list
        	List of most common words indicating the invalidity of a news work.
                
        Returns
        -------
        int
        	The number of trigger words detected within the work.
        
        '''
        for word in keywordList:
            self.keywordCount+=self.content.lower().count(word.lower())
        return self.keywordCount
            
    def printAuthor(self) -> None:
        '''
        Prints the author of the work.
        
        '''
        
        print(self.author)
        return
        
    def printDate(self) -> None:
        '''
        Prints the date of publication of the work.

        '''
        
        print(self.date)    
        return
    
    def read(self) -> None:
        '''
        Prints the entirety of the news work's content for the perusal of the user.
        
        '''
        
        print(self.content)
        return
    
    def dubRatio(self) -> float:
        '''
        Gives the ratio of trigger words, considered "dubious", indicative of a false work of news, to the total number of words in the work.
        
        Returns
        -------
        float
        	The ratio of trigger words to total words in the work.
        '''
        
        return self.keywordCount/len(self.content)