from nltk import flatten
from num2words import num2words

def get_text_from_number(text: str, label: str, language: str) -> str:
        '''
            to input the text and detect if any digits exists, if there is, will convert the numbers into its word representation

            text : obtains an entry of the annotations to do the text preprocessing
        '''

        # split sentence to list of words
        text_list = text.split()

        # append the preprocessed text in this list
        new_text_list = []
        
        for txt in text_list:
            
            # check if word is STRICTLY alphanumeric, not either one of it
            if (txt.isalnum()) and (not txt.isalpha()) and (not txt.isnumeric()):
                sep_alpha_numeric_list = []
                
                # iterate at the letter/digits level
                for letter_idx, letter in enumerate(list(txt)):
                    
                    # append original letter
                    sep_alpha_numeric_list.append(txt[letter_idx])
                    
                    # compare the current indexed letter/digits and the next index letter/digits
                    if letter_idx != len(list(txt))-1 and ((txt[letter_idx].isalpha() and txt[letter_idx+1].isnumeric()) or (txt[letter_idx].isnumeric() and txt[letter_idx+1].isalpha())):                    
                        sep_alpha_numeric_list.append('%')
                
                # join the list of characters to a word again but with '%' separator
                new_text_list.append(''.join(sep_alpha_numeric_list))
                
            # if word is not STRICTLY alphanumeric, just append the word
            else:
                new_text_list.append(txt)
        
        # remove the separator '%'
        preprocessed_text = ' '.join(flatten(new_text_list)).replace('%', ' ')
        
        # split the text into list of words again
        preprocessed_text_list = preprocessed_text.split()
        
        # check and see if the individual strings are digits or not
        for idx, t in enumerate(preprocessed_text_list):
            try:
                if label == 'magister':
                    # if less than 100, will pronounced in the usual way => e.g 55 == fifty-five
                    if float(t) <= 100:
                        preprocessed_text_list[idx] = num2words(t)
                    # else pronounced by its individual number => e.g 119 => one one nine
                    else:
                        sep_num_list = []
                        for k in list(t):
                            sep_num_list.append(num2words(k))
                        preprocessed_text_list[idx] = sep_num_list
                else:
                    if language == 'ms':
                        language = 'id' # no direct translation for ms
                        preprocessed_text_list[idx] = num2words(t, lang=language)
                        # convert bahasa indo to bahasa melayu for the number
                        if preprocessed_text_list[idx] == 'nol':
                            preprocessed_text_list[idx] = 'kosong'
                        elif preprocessed_text_list[idx] == 'delapan':
                            preprocessed_text_list[idx] = 'lapan'
                    else:
                        preprocessed_text_list[idx] = num2words(t, lang=language)
            except:
                continue
                
        # make lists of lists into just a list of words
        text_list_flat = flatten(preprocessed_text_list)

        # returns the preprocessed text, where all the numbers in annotations are converted to text
        return ' '.join(text_list_flat)