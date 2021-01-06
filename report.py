#-----------------------------------------------------------------------------
# Name:        report.py
# Purpose:     To store the Report class.                   
# Author:      Jason Tian
# Created:     05-Jun-2020
# Updated:     05-Jun-2020 
#-----------------------------------------------------------------------------

from newsitem import NewsItem

class Report(NewsItem):
    '''
    A child class of the NewsItem class.
    
    The Report class includes news works that are specifically unbiased and are intended to only represent the facts of a story.
    They are often longer and more thorough than other news works, such as articles, are, but do not argue a particular side or 
    convey the author's opinions. 
    
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
    accuracy : float
    	Decimal number between 0.00 and 1.00 indicating how factually correct the report is.
        The closer the number is to 1.00, the more accurate it is.
    
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
    credibilityCheck() -> bool
    	Returns whether or not the report can be considered "credible" based on an internal check.
    '''
    
    def __init__(self, title, content, valid, accuracy, keywordCount=0, author="?", org="?", topic="?", date="?"):
        '''
        Constructor to build a Report object.
        
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
        accuracy : float
    		Percentage number indicating how factually correct the report is.
        '''                
        super().__init__(title, content, valid, keywordCount, author, org, topic, date)
        self.acc=accuracy
        
    def credibilityCheck(self) -> bool:
        '''
        Determines whether or not a work is credible by comparing its ratio of dubious words to length and its accuracy.
        
        Returns
        -------
        bool
            True if the report's "dubious ratio" is less than its accuracy
            False otherwise
	'''
        if self.dubRatio()<self.acc:
            return True
        else:
            return False