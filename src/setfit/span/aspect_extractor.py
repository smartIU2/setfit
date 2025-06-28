import logging
from typing import TYPE_CHECKING, Optional, List, Tuple, Union


if TYPE_CHECKING:
    from spacy.tokens import Doc


class Aspect():
    """ Used to store two slices of a spacy doc
      , plus polarity label and aspect categories
    """

    MAX_EXPANSION_LEFT = 2
    MAX_EXPANSION_RIGHT = 3

    def __init__(self, doc, start, stop, context_start=None, context_stop=None):
    
        self._reduced_slice = slice(start,stop)
        
        self.doc = doc
        self._expansion_left = 0
        self._expansion_right = 0
        
        self._context_start = context_start
        self._context_stop = context_stop
        
        self.ordinal = 0
        self.label = ''
        self.categories = set()

    @property
    def start(self):
        """ start of reduced slice"""
        return self._reduced_slice.start
          
    @property
    def stop(self):
        """ stop of reduced slice"""
        return self._reduced_slice.stop

    @property
    def context_start(self):
        """ start of slice with context"""
        return self.start if self._context_start is None else self._context_start
        
    @property
    def context_stop(self):
        """ stop of slice with context"""
        return self.stop if self._context_stop is None else self._context_stop
    
    def expand(self):
        """ expand context of span
        
        Returns:
            True, if expansion succesful
            False, if expansion not possible
        """
        
        if (self.context_start > 0) and (self._expansion_left < self.MAX_EXPANSION_LEFT) and (self.doc[self.context_start - 1].text not in (',',';')):
            self._context_start = self.context_start - 1
            self._expansion_left += 1
            return True
            
        if (self.context_stop < len(self.doc)) and (self._expansion_right < self.MAX_EXPANSION_RIGHT) and (self.doc[self.context_stop].text not in (',',';')):
            self._context_stop = self.context_stop + 1
            self._expansion_right += 1
            return True
            
        return False


    @property
    def text(self):
        """ reduced text """
        return self.doc[self.start:self.stop].text
        
    @property
    def context(self):
        """ text from slice with context """
        return self.doc[self.context_start:self.context_stop].text
    
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        
        if isinstance(other, Aspect):
            return self.context == other.context
            
        if isinstance(other, str):
            return self.__str__() == other

        return False


class AspectExtractor:

    """ tokens to include, even though they are not nouns"""
    NON_NOUN_ASPECTS = ('acted', 'directed', 'edited', 'emotionally', 'filmed', 'visually', 'written')

    """ keep these in front of / between extracted nouns """
    CHUNK_EXCEPTIONS = ('emotional', 'musical', 'visual', 'cinematic', 'generated', 'set', 'special', 'comic'
                      , 'another', 'other'
                      , "'s", '-', '/', ',', 'and')

    """ do not keep these in front of extracted nouns """
    CHUNK_STOP_WORDS = ('level', 'minute', 'minutes')

    """ allow 'this' plus optional adjectives in front of these words only """
    MOVIE_SYNONYMS = ('entry', 'flick', 'film', 'mess', 'movie', 'installment', 'version')
    
    """ allowed POS tags """
    POS_WHITELIST = ('NOUN') # note that PROPN is not included, since they should be replaced in the text
    
    """ use this as a noun, _only_ if there are no other aspects in the sentence """
    NOUN_SUBSTITUTE = 'overall'


    def __init__(self
               , spacy_model: Union[str, object]
               , spacy_disable_pipes: Optional[List[str]] = []) -> None:
        super().__init__()
        
        if isinstance(spacy_model, str):
            import spacy
            
            if spacy_model.endswith('trf'):
                spacy.prefer_gpu()
            
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = spacy_model
            
        self.disabled_pipes = spacy_disable_pipes

    
    def _reduce_noun_chunk(self, doc, start, stop):
        """ reduce chunk, unless the chunk equals 'this ___ movie/film/flick'
        
        Returns:
            context_slice: noun chunk stripped of adjectives, adverbs, etc. at the beginning, for aspect model
                         + complete noun chunk, for polarity model
        """

        full_start = start
        for i in range(start, stop):
            # spacy sometimes includes random stuff like "2: " at the beginning of chunks
            if (doc[i].pos_ not in ('X','PUNCT')) and (doc[i].lower_ != "'s"):
                full_start = i
                break
                       
        #TODO: split chunks by comma, if there are multiple nouns; but not if it's a list of adjectives     
        
        reduced_start = full_start
        for i in range(stop - 1, full_start - 1, -1):
            if (doc[i].lower_ in self.MOVIE_SYNONYMS) and (doc[full_start].lower_ == 'this'):
                break
                
            if (doc[i].lower_ in self.CHUNK_STOP_WORDS) or ((doc[i].pos_ not in self.POS_WHITELIST) and (doc[i].lower_ not in self.CHUNK_EXCEPTIONS)):
            
                if i == stop - 1:
                    return None
            
                # clean start (spacy sometimes creates chunks like "clean-cut editing" with "cut" tagged as noun)
                if doc[i+1].text in ("'s", '-', '/', ',', 'and'):
                    reduced_start = min(i + 3, stop - 1)
                else:
                    reduced_start = i + 1
                    
                break

        return Aspect(doc, reduced_start, stop, context_start=full_start, context_stop=stop)


    def __call__(self, texts: List[str]) -> Tuple[List["Doc"], List[Aspect]]:
        aspects_list = []
        docs = list(self.nlp.pipe(texts, disable=self.disabled_pipes))
        for doc in docs:
            
            # collect aspect chunks - note that doc.noun_chunks is purposely not used, as its results are even more erratic than token.left_edge
            min_pos = len(doc)
            aspects = []
            for i in range(len(doc) - 1, -1, -1):
                if i >= min_pos:
                    continue
            
                word = doc[i]
                
                # noun chunk
                if (word.pos_ in self.POS_WHITELIST):
                    chunk = self._reduce_noun_chunk(doc, word.left_edge.i, i + 1)
                    if chunk and chunk != '':                 
                        aspects.insert(0, chunk)
                        min_pos = chunk.context_start
                        
                # whitelisted non-noun aspect
                elif (word.lower_ in self.NON_NOUN_ASPECTS):
                    aspects.insert(0, Aspect(doc, i, i+1))
                    min_pos = i
                
            # join chunks back together when spacy decided to split, e.g., "sub-plot" into three separate chunks
            for i in range(len(aspects) - 1, 0, -1):
         
                if (doc[aspects[i-1].stop - 1].whitespace_ == '') and (aspects[i-1].stop == aspects[i].start):

                    aspects[i-1] = Aspect(doc, aspects[i-1].start, aspects[i].stop
                                           , context_start=aspects[i-1].context_start
                                           , context_stop=aspects[i].context_stop)
                    
                    del aspects[i]
                    
            # substitute for sentences without other aspects
            if len(aspects) == 0:
                aspects = [Aspect(doc, token.i, token.i + 1) for token in doc
                        if token.lower_ == self.NOUN_SUBSTITUTE]
            else:
                # set ordinal
                # and expand context for polarity model, if an aspect is found more than once
                for i in range(len(aspects) - 1):
                    for j in range(i + 1, len(aspects)):
                    
                        if aspects[i].text == aspects[j].text:
                            aspects[j].ordinal = aspects[i].ordinal + 1
                            
                            while aspects[i] == aspects[j]:
                                e1 = aspects[i].expand()
                                e2 = aspects[j].expand()
                                
                                if not (e1 or e2):
                                    logging.warning(f'Could not expand possible aspect {i} "{aspects[i].text}" in "{doc.text}" to remove ambiguity.')
                                    #TODO: remove duplicate (?)
                                    break

            aspects_list.append(aspects)
            
        return docs, aspects_list