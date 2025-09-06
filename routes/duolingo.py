# -*- coding: utf-8 -*-
import os
import re
from typing import List, Tuple
import logging
from flask import request, jsonify
from routes import app
from flask import Flask, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)

# --- Config / logging ---
logging.basicConfig(level=logging.INFO)



from flask import Flask, request, jsonify
import re
import logging


logging.basicConfig(level=logging.DEBUG)

class NumberParser:
    def __init__(self):
        # Roman numeral mapping with validation
        self.roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
        # Extended English number words
        self.english_ones = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'a': 1, 'an': 1  # Handle "a hundred", "an hour"
        }
        
        self.english_tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        
        self.english_scales = {
            'hundred': 100, 'thousand': 1000, 'million': 1000000, 
            'billion': 1000000000, 'trillion': 1000000000000
        }
        
        # Extended German number words
        self.german_ones = {
            'null': 0, 'ein': 1, 'eine': 1, 'eins': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 
            'fünf': 5, 'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
            'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
            'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19
        }
        
        self.german_tens = {
            'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50,
            'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90
        }
        
        self.german_scales = {
            'hundert': 100, 'tausend': 1000, 'million': 1000000, 
            'milliarde': 1000000000, 'billion': 1000000000000
        }
        
        # Comprehensive Chinese number mappings
        self.chinese_digits = {
            # Traditional Chinese
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            # Simplified Chinese (same as traditional for basic digits)
            '〇': 0, '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
            # Alternative forms
            '两': 2, '兩': 2  # Alternative for "two"
        }
        
        self.chinese_units = {
            # Traditional Chinese
            '十': 10, '百': 100, '千': 1000, '萬': 10000, '億': 100000000, '兆': 1000000000000,
            # Simplified Chinese  
            '拾': 10, '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000,
            # Alternative forms
            '什': 10  # Alternative for ten
        }
        
        # French numbers (additional language support)
        self.french_ones = {
            'zéro': 0, 'un': 1, 'une': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
            'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10,
            'onze': 11, 'douze': 12, 'treize': 13, 'quatorze': 14, 'quinze': 15,
            'seize': 16, 'dix-sept': 17, 'dix-huit': 18, 'dix-neuf': 19
        }
        
        self.french_tens = {
            'vingt': 20, 'trente': 30, 'quarante': 40, 'cinquante': 50,
            'soixante': 60, 'quatre-vingt': 80, 'quatre-vingts': 80
        }
        
        # Spanish numbers (additional language support)
        self.spanish_ones = {
            'cero': 0, 'uno': 1, 'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
            'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
            'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
            'dieciséis': 16, 'diecisiete': 17, 'dieciocho': 18, 'diecinueve': 19
        }

    def is_valid_roman(self, s):
        """Validate Roman numeral format"""
        s = s.upper()
        # Check for valid Roman numeral pattern
        pattern = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
        return bool(re.match(pattern, s))

    def parse_roman(self, s):
        """Parse Roman numeral to integer with validation"""
        if not self.is_valid_roman(s):
            return None
            
        s = s.upper()
        total = 0
        prev_value = 0
        
        for char in reversed(s):
            value = self.roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        # Additional validation - check if result is in valid range
        return total if 1 <= total <= 3999 else None

    def parse_english(self, s):
        """Parse English number words to integer with enhanced support"""
        s = s.lower().strip().replace('-', ' ').replace(',', '')
        
        # Handle "and" in numbers like "one hundred and twenty"
        s = s.replace(' and ', ' ')
        
        # Handle simple cases
        if s in self.english_ones:
            return self.english_ones[s]
            
        # Split by spaces and process
        words = s.split()
        total = 0
        current = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word in self.english_ones:
                current += self.english_ones[word]
            elif word in self.english_tens:
                current += self.english_tens[word]
            elif word in self.english_scales:
                if word == 'hundred':
                    if current == 0:  # Handle "hundred" without preceding number
                        current = 100
                    else:
                        current *= 100
                elif word in ['thousand', 'million', 'billion', 'trillion']:
                    if current == 0:  # Handle "thousand" without preceding number
                        current = 1
                    total += current * self.english_scales[word]
                    current = 0
            i += 1
        
        return total + current

    def parse_german_compound(self, s):
        """Parse German compound numbers like einundzwanzig, dreihundertvierzehn"""
        s = s.lower().strip()
        
        # Handle numbers with "und" (like einundzwanzig = 21)
        if 'und' in s:
            parts = s.split('und')
            if len(parts) == 2:
                ones_part = parts[0]
                tens_part = parts[1]
                
                ones_val = self.german_ones.get(ones_part, 0)
                tens_val = self.german_tens.get(tens_part, 0)
                
                if ones_val and tens_val:
                    return tens_val + ones_val
        
        # Handle hundreds (like dreihundert = 300, dreihundertvierzehn = 314)
        total = 0
        remaining = s
        
        # Check for hundreds
        for word, value in self.german_ones.items():
            if remaining.startswith(word + 'hundert'):
                total += value * 100
                remaining = remaining[len(word + 'hundert'):]
                break
        
        if remaining.startswith('hundert') and total == 0:
            total = 100
            remaining = remaining[7:]
        
        # Handle remaining part after hundreds
        if remaining:
            if 'und' in remaining:
                parts = remaining.split('und')
                if len(parts) == 2:
                    ones_val = self.german_ones.get(parts[0], 0)
                    tens_val = self.german_tens.get(parts[1], 0)
                    if ones_val and tens_val:
                        total += tens_val + ones_val
                        return total
            
            # Check for simple remaining numbers
            if remaining in self.german_ones:
                total += self.german_ones[remaining]
            elif remaining in self.german_tens:
                total += self.german_tens[remaining]
        
        return total if total > 0 else None

    def parse_german(self, s):
        """Parse German number words to integer with comprehensive support"""
        s = s.lower().strip()
        
        # Try compound parsing first
        result = self.parse_german_compound(s)
        if result is not None:
            return result
            
        # Handle simple numbers
        if s in self.german_ones:
            return self.german_ones[s]
        if s in self.german_tens:
            return self.german_tens[s]
        if s in self.german_scales:
            return self.german_scales[s]
            
        return None

    def parse_chinese(self, s):
        """Parse Chinese numerals with enhanced support for edge cases"""
        total = 0
        current = 0
        temp = 0
        
        # Handle special case of just "十" or "拾" (means 10)
        if s in ['十', '拾']:
            return 10
            
        i = 0
        while i < len(s):
            char = s[i]
            
            if char in self.chinese_digits:
                temp = self.chinese_digits[char]
            elif char in self.chinese_units:
                unit_value = self.chinese_units[char]
                
                if unit_value >= 10000:  # 萬, 億, etc.
                    if temp == 0 and current == 0:
                        temp = 1  # Handle cases like "萬" meaning "one 萬"
                    total = (total + current + temp) * unit_value
                    current = 0
                    temp = 0
                elif unit_value >= 10:  # 十, 百, 千
                    if temp == 0:
                        temp = 1  # Handle cases like "十" meaning "one 十"
                    current += temp * unit_value
                    temp = 0
                else:
                    current += temp
                    temp = 0
            i += 1
            
        return total + current + temp

    def parse_french(self, s):
        """Basic French number parsing"""
        s = s.lower().strip().replace('-', ' ')
        
        if s in self.french_ones:
            return self.french_ones[s]
        if s in self.french_tens:
            return self.french_tens[s]
            
        # Handle some compound numbers
        words = s.split()
        total = 0
        current = 0
        
        for word in words:
            if word in self.french_ones:
                current += self.french_ones[word]
            elif word in self.french_tens:
                current += self.french_tens[word]
            elif word == 'cent':
                current *= 100
            elif word == 'mille':
                total += current * 1000
                current = 0
                
        return total + current

    def parse_spanish(self, s):
        """Basic Spanish number parsing"""
        s = s.lower().strip()
        
        if s in self.spanish_ones:
            return self.spanish_ones[s]
            
        return None

    def detect_and_parse(self, s):
        """Detect language and parse number with comprehensive edge case handling"""
        original_s = s
        s = s.strip()
        
        # Handle empty or whitespace-only strings
        if not s:
            return 0, 'unknown'
        
        # Try Arabic numeral first (including negative numbers and decimals)
        try:
            # Handle decimals by taking integer part
            if '.' in s:
                return int(float(s)), 'arabic'
            return int(s), 'arabic'
        except ValueError:
            pass
        
        # Try Roman numeral with strict validation
        if re.match(r'^[IVXLCDM]+$', s.upper()):
            result = self.parse_roman(s)
            if result is not None:
                return result, 'roman'
        
        # Try Chinese (contains Chinese characters)
        if any(c in self.chinese_digits or c in self.chinese_units for c in s):
            try:
                result = self.parse_chinese(s)
                return result, 'chinese'
            except:
                pass
        
        # Language detection based on specific words/patterns
        s_lower = s.lower()
        
        # German detection (more comprehensive)
        german_indicators = ['und', 'hundert', 'tausend', 'zig', 'zehn', 'elf', 'zwölf']
        if any(indicator in s_lower for indicator in german_indicators):
            result = self.parse_german(s)
            if result is not None:
                return result, 'german'
        
        # French detection
        french_indicators = ['vingt', 'trente', 'quarante', 'soixante', 'quatre', 'cent', 'mille']
        if any(indicator in s_lower for indicator in french_indicators):
            result = self.parse_french(s)
            if result is not None:
                return result, 'french'
        
        # Spanish detection
        spanish_indicators = ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez']
        if any(indicator in s_lower for indicator in spanish_indicators):
            result = self.parse_spanish(s)
            if result is not None:
                return result, 'spanish'
        
        # Try English with enhanced parsing
        try:
            result = self.parse_english(s)
            if result is not None:
                return result, 'english'
        except:
            pass
        
        # Try German again as fallback (for compound numbers)
        result = self.parse_german(s)
        if result is not None:
            return result, 'german'
        
        # Log unknown formats for debugging
        app.logger.warning(f"Unknown number format: '{original_s}'")
        return 0, 'unknown'

# def get_language_priority(lang):
#     """Get sorting priority for languages when values are equal"""
#     priority_map = {
#         'roman': 0,
#         'english': 1, 
#         'chinese_traditional': 2,
#         'chinese_simplified': 3,
#         'german': 4,
#         'arabic': 5,
#         'french': 6,
#         'spanish': 7,
#         'unknown': 8
#     }
#     return priority_map.get(lang, 9)

def get_language_priority(lang):
    """Get sorting priority for languages when values are equal"""
    priority_map = {
        'roman': 0,
        'english': 1,
        'chinese_traditional': 2,
        'chinese_simplified': 3,
        'german': 4,
        'arabic': 5
    }
    # Default to lowest priority if unknown
    return priority_map.get(lang, 6)



def detect_chinese_type(s):
    """Detect if Chinese text is traditional or simplified"""
    logger.info(f"🔍 Detecting Chinese type for: '{s}'")
    
    traditional_chars = set('萬億兆')
    simplified_chars = set('万亿')
    
    traditional_found = [c for c in s if c in traditional_chars]
    simplified_found = [c for c in s if c in simplified_chars]
    
    if traditional_found:
        logger.info(f"   Traditional chars found: {traditional_found} -> chinese_traditional")
        return 'chinese_traditional'
    elif simplified_found:
        logger.info(f"   Simplified chars found: {simplified_found} -> chinese_simplified")
        return 'chinese_simplified'
    else:
        # Use character frequency analysis for ambiguous cases
        traditional_indicators = set('數龍國門開關東車無書學時間問題')
        simplified_indicators = set('数龙国门开关东车无书学时间问题')
        
        trad_indicators = [c for c in s if c in traditional_indicators]
        simp_indicators = [c for c in s if c in simplified_indicators]
        
        if trad_indicators:
            logger.info(f"   Traditional indicators found: {trad_indicators} -> chinese_traditional")
            return 'chinese_traditional'
        elif simp_indicators:
            logger.info(f"   from flask import Flask, request, jsonify")



logging.basicConfig(level=logging.DEBUG)

class NumberParser:
    def __init__(self):
        # Roman numeral mapping with validation
        self.roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
        # Extended English number words
        self.english_ones = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'a': 1, 'an': 1  # Handle "a hundred", "an hour"
        }
        
        self.english_tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        
        self.english_scales = {
            'hundred': 100, 'thousand': 1000, 'million': 1000000, 
            'billion': 1000000000, 'trillion': 1000000000000
        }
        
        # Extended German number words
        self.german_ones = {
            'null': 0, 'ein': 1, 'eine': 1, 'eins': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 
            'fünf': 5, 'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
            'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
            'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19
        }
        
        self.german_tens = {
            'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50,
            'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90
        }
        
        self.german_scales = {
            'hundert': 100, 'tausend': 1000, 'million': 1000000, 
            'milliarde': 1000000000, 'billion': 1000000000000
        }
        
        # Comprehensive Chinese number mappings
        self.chinese_digits = {
            # Traditional Chinese
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            # Simplified Chinese (same as traditional for basic digits)
            '〇': 0, '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
            # Alternative forms
            '两': 2, '兩': 2  # Alternative for "two"
        }
        
        self.chinese_units = {
            # Traditional Chinese
            '十': 10, '百': 100, '千': 1000, '萬': 10000, '億': 100000000, '兆': 1000000000000,
            # Simplified Chinese  
            '拾': 10, '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000,
            # Alternative forms
            '什': 10  # Alternative for ten
        }
        
        # French numbers (additional language support)
        self.french_ones = {
            'zéro': 0, 'un': 1, 'une': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
            'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10,
            'onze': 11, 'douze': 12, 'treize': 13, 'quatorze': 14, 'quinze': 15,
            'seize': 16, 'dix-sept': 17, 'dix-huit': 18, 'dix-neuf': 19
        }
        
        self.french_tens = {
            'vingt': 20, 'trente': 30, 'quarante': 40, 'cinquante': 50,
            'soixante': 60, 'quatre-vingt': 80, 'quatre-vingts': 80
        }
        
        # Spanish numbers (additional language support)
        self.spanish_ones = {
            'cero': 0, 'uno': 1, 'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
            'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
            'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
            'dieciséis': 16, 'diecisiete': 17, 'dieciocho': 18, 'diecinueve': 19
        }

    def is_valid_roman(self, s):
        """Validate Roman numeral format"""
        s = s.upper()
        # Check for valid Roman numeral pattern
        pattern = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
        return bool(re.match(pattern, s))

    def parse_roman(self, s):
        """Parse Roman numeral to integer with validation"""
        if not self.is_valid_roman(s):
            return None
            
        s = s.upper()
        total = 0
        prev_value = 0
        
        for char in reversed(s):
            value = self.roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        # Additional validation - check if result is in valid range
        return total if 1 <= total <= 3999 else None

    def parse_english(self, s):
        """Parse English number words to integer with enhanced support"""
        s = s.lower().strip().replace('-', ' ').replace(',', '')
        
        # Handle "and" in numbers like "one hundred and twenty"
        s = s.replace(' and ', ' ')
        
        # Handle simple cases
        if s in self.english_ones:
            return self.english_ones[s]
            
        # Split by spaces and process
        words = s.split()
        total = 0
        current = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word in self.english_ones:
                current += self.english_ones[word]
            elif word in self.english_tens:
                current += self.english_tens[word]
            elif word in self.english_scales:
                if word == 'hundred':
                    if current == 0:  # Handle "hundred" without preceding number
                        current = 100
                    else:
                        current *= 100
                elif word in ['thousand', 'million', 'billion', 'trillion']:
                    if current == 0:  # Handle "thousand" without preceding number
                        current = 1
                    total += current * self.english_scales[word]
                    current = 0
            i += 1
        
        return total + current

    def parse_german_compound(self, s):
        """Parse German compound numbers like einundzwanzig, dreihundertvierzehn"""
        s = s.lower().strip()
        
        # Handle numbers with "und" (like einundzwanzig = 21)
        if 'und' in s:
            parts = s.split('und')
            if len(parts) == 2:
                ones_part = parts[0]
                tens_part = parts[1]
                
                ones_val = self.german_ones.get(ones_part, 0)
                tens_val = self.german_tens.get(tens_part, 0)
                
                if ones_val and tens_val:
                    return tens_val + ones_val
        
        # Handle hundreds (like dreihundert = 300, dreihundertvierzehn = 314)
        total = 0
        remaining = s
        
        # Check for hundreds
        for word, value in self.german_ones.items():
            if remaining.startswith(word + 'hundert'):
                total += value * 100
                remaining = remaining[len(word + 'hundert'):]
                break
        
        if remaining.startswith('hundert') and total == 0:
            total = 100
            remaining = remaining[7:]
        
        # Handle remaining part after hundreds
        if remaining:
            if 'und' in remaining:
                parts = remaining.split('und')
                if len(parts) == 2:
                    ones_val = self.german_ones.get(parts[0], 0)
                    tens_val = self.german_tens.get(parts[1], 0)
                    if ones_val and tens_val:
                        total += tens_val + ones_val
                        return total
            
            # Check for simple remaining numbers
            if remaining in self.german_ones:
                total += self.german_ones[remaining]
            elif remaining in self.german_tens:
                total += self.german_tens[remaining]
        
        return total if total > 0 else None

    def parse_german(self, s):
        """Parse German number words to integer with comprehensive support"""
        s = s.lower().strip()
        
        # Try compound parsing first
        result = self.parse_german_compound(s)
        if result is not None:
            return result
            
        # Handle simple numbers
        if s in self.german_ones:
            return self.german_ones[s]
        if s in self.german_tens:
            return self.german_tens[s]
        if s in self.german_scales:
            return self.german_scales[s]
            
        return None

    def parse_chinese(self, s):
        """Parse Chinese numerals with enhanced support for edge cases"""
        total = 0
        current = 0
        temp = 0
        
        # Handle special case of just "十" or "拾" (means 10)
        if s in ['十', '拾']:
            return 10
            
        i = 0
        while i < len(s):
            char = s[i]
            
            if char in self.chinese_digits:
                temp = self.chinese_digits[char]
            elif char in self.chinese_units:
                unit_value = self.chinese_units[char]
                
                if unit_value >= 10000:  # 萬, 億, etc.
                    if temp == 0 and current == 0:
                        temp = 1  # Handle cases like "萬" meaning "one 萬"
                    total = (total + current + temp) * unit_value
                    current = 0
                    temp = 0
                elif unit_value >= 10:  # 十, 百, 千
                    if temp == 0:
                        temp = 1  # Handle cases like "十" meaning "one 十"
                    current += temp * unit_value
                    temp = 0
                else:
                    current += temp
                    temp = 0
            i += 1
            
        return total + current + temp

    def parse_french(self, s):
        """Basic French number parsing"""
        s = s.lower().strip().replace('-', ' ')
        
        if s in self.french_ones:
            return self.french_ones[s]
        if s in self.french_tens:
            return self.french_tens[s]
            
        # Handle some compound numbers
        words = s.split()
        total = 0
        current = 0
        
        for word in words:
            if word in self.french_ones:
                current += self.french_ones[word]
            elif word in self.french_tens:
                current += self.french_tens[word]
            elif word == 'cent':
                current *= 100
            elif word == 'mille':
                total += current * 1000
                current = 0
                
        return total + current

    def parse_spanish(self, s):
        """Basic Spanish number parsing"""
        s = s.lower().strip()
        
        if s in self.spanish_ones:
            return self.spanish_ones[s]
            
        return None

    def detect_and_parse(self, s):
        """Detect language and parse number with comprehensive edge case handling"""
        original_s = s
        s = s.strip()
        
        # Handle empty or whitespace-only strings
        if not s:
            return 0, 'unknown'
        
        # Try Arabic numeral first (including negative numbers and decimals)
        try:
            # Handle decimals by taking integer part
            if '.' in s:
                return int(float(s)), 'arabic'
            return int(s), 'arabic'
        except ValueError:
            pass
        
        # Try Roman numeral with strict validation
        if re.match(r'^[IVXLCDM]+$', s.upper()):
            result = self.parse_roman(s)
            if result is not None:
                return result, 'roman'
        
        # Try Chinese (contains Chinese characters)
        if any(c in self.chinese_digits or c in self.chinese_units for c in s):
            try:
                result = self.parse_chinese(s)
                return result, 'chinese'
            except:
                pass
        
        # Language detection based on specific words/patterns
        s_lower = s.lower()
        
        # German detection (more comprehensive)
        german_indicators = ['und', 'hundert', 'tausend', 'zig', 'zehn', 'elf', 'zwölf']
        if any(indicator in s_lower for indicator in german_indicators):
            result = self.parse_german(s)
            if result is not None:
                return result, 'german'
        
        # French detection
        french_indicators = ['vingt', 'trente', 'quarante', 'soixante', 'quatre', 'cent', 'mille']
        if any(indicator in s_lower for indicator in french_indicators):
            result = self.parse_french(s)
            if result is not None:
                return result, 'french'
        
        # Spanish detection
        spanish_indicators = ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez']
        if any(indicator in s_lower for indicator in spanish_indicators):
            result = self.parse_spanish(s)
            if result is not None:
                return result, 'spanish'
        
        # Try English with enhanced parsing
        try:
            result = self.parse_english(s)
            if result is not None:
                return result, 'english'
        except:
            pass
        
        # Try German again as fallback (for compound numbers)
        result = self.parse_german(s)
        if result is not None:
            return result, 'german'
        
        # Log unknown formats for debugging
        app.logger.warning(f"Unknown number format: '{original_s}'")
        return 0, 'unknown'

def get_language_priority(lang):
    """Get sorting priority for languages when values are equal"""
    priority_map = {
        'roman': 0,
        'english': 1, 
        'chinese_traditional': 2,
        'chinese_simplified': 3,
        'german': 4,
        'arabic': 5,
        'french': 6,
        'spanish': 7,
        'unknown': 8
    }
    return priority_map.get(lang, 9)

def detect_chinese_type(s):
    """Detect if Chinese text is traditional or simplified"""
    traditional_chars = set('萬億兆')
    simplified_chars = set('万亿')
    
    if any(c in traditional_chars for c in s):
        return 'chinese_traditional'
    elif any(c in simplified_chars for c in s):
        return 'chinese_simplified'
    else:
        # Use character frequency analysis for ambiguous cases
        # Most common traditional vs simplified differences
        traditional_indicators = set('數龍國門開關東車無書學時間問題')
        simplified_indicators = set('数龙国门开关东车无书学时间问题')
        
        if any(c in traditional_indicators for c in s):
            return 'chinese_traditional'
        elif any(c in simplified_indicators for c in s):
            return 'chinese_simplified'
        
        # Default to traditional for ambiguous cases
        return 'chinese_traditional'

@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    try:
        data = request.get_json()
        # --- log the raw body + parsed fields for each test case ---
        raw_body = request.get_data(cache=False, as_text=True)
        challenge = (data or {}).get('challenge')  # Duolingo payload includes this
        part = (data or {}).get('part')
        app.logger.info("📝 /duolingo-sort request | challenge=%s | part=%s | body=%s",
                        challenge, part, raw_body)
        
        # Validate input structure
        if not data or 'part' not in data or 'challengeInput' not in data:
            return jsonify({"error": "Invalid input structure"}), 400
            
        part = data.get('part')
        challenge_input = data.get('challengeInput', {})
        unsorted_list = challenge_input.get('unsortedList', [])
        
        # Validate part value
        if part not in ["ONE", "TWO"]:
            return jsonify({"error": "Invalid part value. Must be 'ONE' or 'TWO'"}), 400
        
        # Handle empty list
        if not unsorted_list:
            return jsonify({"sortedList": []})
        
        parser = NumberParser()
        
        # Parse all numbers with their original strings
        parsed_numbers = []
        parsing_errors = []
        
        for i, original_str in enumerate(unsorted_list):
            try:
                if not isinstance(original_str, str):
                    original_str = str(original_str)
                    
                value, lang = parser.detect_and_parse(original_str)
                
                # Refine Chinese language detection
                if lang == 'chinese':
                    lang = detect_chinese_type(original_str)
                
                parsed_numbers.append((value, original_str, lang))
                
                # Log successful parsing for debugging
                app.logger.debug(f"Parsed '{original_str}' -> {value} ({lang})")
                
            except Exception as e:
                parsing_errors.append(f"Error parsing item {i} ('{original_str}'): {str(e)}")
                # Add with value 0 and unknown language as fallback
                parsed_numbers.append((0, original_str, 'unknown'))
        
        # Log parsing errors for debugging
        if parsing_errors:
            app.logger.warning(f"Parsing errors: {parsing_errors}")
        
        if part == "ONE":
            # Part 1: Sort and return as integers
            parsed_numbers.sort(key=lambda x: (x[0], get_language_priority(x[2])))
            sorted_list = [str(num[0]) for num in parsed_numbers]
            
        elif part == "TWO":
            # Part 2: Sort by value, then by language priority, keep original format
            parsed_numbers.sort(key=lambda x: (x[0], get_language_priority(x[2])))
            sorted_list = [num[1] for num in parsed_numbers]
        
        response = {"sortedList": sorted_list}
        
        
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "healthy", "version": "2.0"})

# @app.route('/test', methods=['POST'])
# def test_parser():
#     """Test endpoint to check number parsing"""
#     try:
#         data = request.get_json()
#         test_numbers = data.get('numbers', [])
        
#         parser = NumberParser()
#         results = []
        
#         for num_str in test_numbers:
#             value, lang = parser.detect_and_parse(num_str)
#             if lang == 'chinese':
#                 lang = detect_chinese_type(num_str)
#             results.append({
#                 "original": num_str,
#                 "value": value,
#                 "language": lang,
#                 "priority": get_language_priority(lang)
#             })
        
#         return jsonify({"results": results})
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

