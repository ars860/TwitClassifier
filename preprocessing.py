import re
import unicodedata
from collections import Counter
import nltk
from nltk.corpus import stopwords


class Clean:
    def __init__(self):
        self.regex_dict = {
            'URL': r"""(?xi)\b(?:(?:https?|ftp|file):\/\/|www\.|ftp\.|pic\.|twitter\.|facebook\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:;,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:;,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])""",
            'EMOJI': u'([\U0001F1E0-\U0001F1FF])|([\U0001F300-\U0001F5FF])|([\U0001F600-\U0001F64F])|([\U0001F680-\U0001F6FF])|([\U0001F700-\U0001F77F])|([\U0001F800-\U0001F8FF])|([\U0001F900-\U0001F9FF])|([\U0001FA00-\U0001FA6F])|([\U0001FA70-\U0001FAFF])|([\U00002702-\U000027B0])|([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])',
            'HASHTAG': r"\#\b[\w\-\_]+\b",
            'EMAIL': r"(?:^|(?<=[^\w@.)]))(?:[\w+-](?:\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(?:\.(?:[a-z]{2,})){1,3}(?:$|(?=\b))",
            'MENTION': r"@[A-Za-z0-9]+",
            'CASHTAG': r"(?:[$\u20ac\u00a3\u00a2]\d+(?:[\\.,']\d+)?(?:[MmKkBb](?:n|(?:il(?:lion)?))?)?)|(?:\d+(?:[\\.,']\\d+)?[$\u20ac\u00a3\u00a2])",
            'DATE': r"(?:(?:(?:(?:(?<!:)\b\'?\d{1,4},? ?)?\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b(?:[,\\/]? ?\'?\d{2,4}[a-zA-Z]*)?(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b))|(?:(?:(?<!:)\b\\'?\d{1,4},? ?)\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b(?:[,\\/]? ?\'?\d{2,4}[a-zA-Z]*)?(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b)?))|(?:\b(?<!\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\.\d)\b))",
            'TIME': r'(?:(?:\d+)?\\.?\d+(?:AM|PM|am|pm|a\\.m\\.|p\\.m\\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\\.m\\.|p\\.m\\.))?)',
            'EMPHASIS': r"(?:\*\b\w+\b\*)",
            'ELONG': r"\b[A-Za-z]*([a-zA-Z])\1\1[A-Za-z]*\b"}
        self.regex = self.get_compiled()
        self.contraction_mapping = {"’": "'", "RT ": " ", "ain't": "is not", "aren't": "are not", "can't": "can not",
                                    "'cause": "because", "could've": "could have",
                                    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                                    "don't": "do not", "hadn't": "had not",
                                    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                                    "he's": "he is",
                                    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                                    "how's": "how is", "I'd": "I would",
                                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                                    "I've": "I have",
                                    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                                    "i'll've": "i will have", "i'm": "i am",
                                    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                                    "it'll": "it will",
                                    "it'll've": "it will have", "it's": "it is", "it’s": "it is", "let's": "let us",
                                    "ma'am": "madam", "mayn't": "may not",
                                    "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                                    "must've": "must have",
                                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                                    "needn't've": "need not have",
                                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                                    "shan't": "shall not",
                                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                                    "she'd've": "she would have",
                                    "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                                    "should've": "should have",
                                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                                    "so's": "so as",
                                    "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                                    "that's": "that is",
                                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                                    "here's": "here is",
                                    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                                    "they'll've": "they will have",
                                    "they're": "they are", "they've": "they have", "to've": "to have",
                                    "wasn't": "was not", "we'd": "we would",
                                    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                                    "we're": "we are", "we've": "we have",
                                    "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                                    "what're": "what are",
                                    "what's": "what is", "what've": "what have", "when's": "when is",
                                    "when've": "when have", "where'd": "where did",
                                    "where's": "where is", "where've": "where have", "who'll": "who will",
                                    "who'll've": "who will have",
                                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                                    "will've": "will have",
                                    "won't": "will not", "won't've": "will not have", "would've": "would have",
                                    "wouldn't": "would not",
                                    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                                    "y'all'd've": "you all would have",
                                    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                                    "you'd've": "you would have",
                                    "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                                    "you've": "you have", "It's": "It is", "You'd": "You would",
                                    ' u ': " you ", 'yrs': 'years', 'FYI': 'For your information', ' im ': ' I am ',
                                    'lol': 'LOL', 'You\'re': 'You are'
            , 'can’t': 'can not', '…': '. ', '...': '. ', '\'\'': '\'', '≠': '', 'ain’t': 'am not', 'I’m': 'I am',
                                    'RT\'s': ''}
        self.emoticons = {
            ':*': '<kiss>',
            ':-*': '<kiss>',
            ':x': '<kiss>',
            ':-)': '<happy>',
            ':-))': '<happy>',
            ':-)))': '<happy>',
            ':-))))': '<happy>',
            ':-)))))': '<happy>',
            ':-))))))': '<happy>',
            ':)': '<happy>',
            ':))': '<happy>',
            ':)))': '<happy>',
            ':))))': '<happy>',
            ':)))))': '<happy>',
            ':))))))': '<happy>',
            ':)))))))': '<happy>',
            ':o)': '<happy>',
            ':]': '<happy>',
            ':3': '<happy>',
            ':c)': '<happy>',
            ':>': '<happy>',
            '=]': '<happy>',
            '8)': '<happy>',
            '=)': '<happy>',
            ':}': '<happy>',
            ':^)': '<happy>',
            '|;-)': '<happy>',
            ":'-)": '<happy>',
            ":')": '<happy>',
            '\o/': '<happy>',
            '*\\0/*': '<happy>',
            ':-D': '<laugh>',
            ':D': '<laugh>',
            '8-D': '<laugh>',
            '8D': '<laugh>',
            'x-D': '<laugh>',
            'xD': '<laugh>',
            'X-D': '<laugh>',
            'XD': '<laugh>',
            '=-D': '<laugh>',
            '=D': '<laugh>',
            '=-3': '<laugh>',
            '=3': '<laugh>',
            'B^D': '<laugh>',
            '>:[': '<sad>',
            ':-(': '<sad>',
            ':-((': '<sad>',
            ':-(((': '<sad>',
            ':-((((': '<sad>',
            ':-(((((': '<sad>',
            ':-((((((': '<sad>',
            ':-(((((((': '<sad>',
            ':(': '<sad>',
            ':((': '<sad>',
            ':(((': '<sad>',
            ':((((': '<sad>',
            ':(((((': '<sad>',
            ':((((((': '<sad>',
            ':(((((((': '<sad>',
            ':((((((((': '<sad>',
            ':-c': '<sad>',
            ':c': '<sad>',
            ':-<': '<sad>',
            ':<': '<sad>',
            ':-[': '<sad>',
            ':[': '<sad>',
            ':{': '<sad>',
            ':-||': '<sad>',
            ':@': '<sad>',
            ":'-(": '<sad>',
            ":'(": '<sad>',
            'D:<': '<sad>',
            'D:': '<sad>',
            'D8': '<sad>',
            'D;': '<sad>',
            'D=': '<sad>',
            'DX': '<sad>',
            'v.v': '<sad>',
            "D-':": '<sad>',
            '(>_<)': '<sad>',
            ':|': '<sad>',
            '>:O': '<surprise>',
            ':-O': '<surprise>',
            ':-o': '<surprise>',
            ':O': '<surprise>',
            '°o°': '<surprise>',
            'o_O': '<surprise>',
            'o_0': '<surprise>',
            'o.O': '<surprise>',
            'o-o': '<surprise>',
            '8-0': '<surprise>',
            '|-O': '<surprise>',
            ';-)': '<wink>',
            ';)': '<wink>',
            '*-)': '<wink>',
            '*)': '<wink>',
            ';-]': '<wink>',
            ';]': '<wink>',
            ';D': '<wink>',
            ';^)': '<wink>',
            ':-,': '<wink>',
            '>:P': '<tong>',
            ':-P': '<tong>',
            ':P': '<tong>',
            'X-P': '<tong>',
            'x-p': '<tong>',
            ':-p': '<tong>',
            ':p': '<tong>',
            '=p': '<tong>',
            ':-Þ': '<tong>',
            ':Þ': '<tong>',
            ':-b': '<tong>',
            ':b': '<tong>',
            ':-&': '<tong>',
            '>:\\': '<annoyed>',
            '>:/': '<annoyed>',
            ':-/': '<annoyed>',
            ':-.': '<annoyed>',
            ':/': '<annoyed>',
            ':\\': '<annoyed>',
            '=/': '<annoyed>',
            '=\\': '<annoyed>',
            ':L': '<annoyed>',
            '=L': '<annoyed>',
            ':S': '<annoyed>',
            '>.<': '<annoyed>',
            ':-|': '<annoyed>',
            '<:-|': '<annoyed>',
            ':-X': '<seallips>',
            ':X': '<seallips>',
            ':-#': '<seallips>',
            ':#': '<seallips>',
            'O:-)': '<angel>',
            '0:-3': '<angel>',
            '0:3': '<angel>',
            '0:-)': '<angel>',
            '0:)': '<angel>',
            '0;^)': '<angel>',
            '>:)': '<devil>',
            '>:D': '<devil>',
            '>:-D': '<devil>',
            '>;)': '<devil>',
            '>:-)': '<devil>',
            '}:-)': '<devil>',
            '}:)': '<devil>',
            '3:-)': '<devil>',
            '3:)': '<devil>',
            'o/\o': '<highfive>',
            '^5': '<highfive>',
            '>_>^': '<highfive>',
            '^<_<': '<highfive>',
            '<3': '<heart>',
            '^3^': '<smile>',
            "(':": '<smile>',
            " > < ": '<smile>',
            "UvU": '<smile>',
            "uwu": '<smile>',
            'UwU': '<smile>'
        }

    def get_compiled(self):
        regexes = {k: re.compile(self.regex_dict[k]) for k, v in
                   self.regex_dict.items()}
        return regexes

    def __call__(self, sentence):
        for key, reg in self.regex.items():
            sentence = self.regex[key].sub(lambda m: " <" + key + "> ", sentence)
        for word in self.emoticons.keys():
            sentence = sentence.replace(word, self.emoticons[word])
        sentence = sentence.lower()
        for word in self.contraction_mapping.keys():
            sentence = sentence.replace(word, self.contraction_mapping[word])
        sentence = re.sub(r"[\-\"`@#$%^&*(|)/~\[\]{}:;+,._='!?]+", " ", sentence)
        sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', errors='ignore').decode('utf8',
                                                                                                   errors='ignore')
        # sentence = re.sub(r'\b([b-hB-Hj-zJ-Z] )', ' ', sentence)
        # sentence = re.sub(r'( [b-hB-Hj-zJ-Z])\b', ' ', sentence)
        # sentence = re.sub(r'((istj|istp|isfj|isfp|infj|infp|intj|intp|estp|estj|esfp|esfj|enfp|enfj|entp|entj))',
        #                  '<type>', sentence, flags=re.IGNORECASE)

        sentence = list(filter(lambda s: len(s) > 0, sentence.split()))
        return sentence


class StopWords:
    def __init__(self, sentences, stemmer, stops_cnt=20):
        nltk.download('stopwords')

        text_words = Counter([word for sentence in sentences for word in sentence])
        text_words = sorted([(word, cnt) for word, cnt in text_words.items()], key=lambda wc: -wc[1])
        text_words = [word for word, cnt in text_words]
        text_words = list(filter(lambda w: not (w.startswith('<') and w.endswith('>')), text_words))
        text_words = set(text_words[:stops_cnt])
        nltk_words = set(map(stemmer.stemWord, stopwords.words('english')))
        self.stop_words = nltk_words # text_words |

    def __call__(self, sentence):
        return list(filter(lambda word: word not in self.stop_words, sentence))
