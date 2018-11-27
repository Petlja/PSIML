import wikipedia
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

# List of languages given with language codes.
languages = ["ko", "el", "en", "sr"]

# List of search terms to be used to obtain text from Wikipedia.
terms_by_language = {
    "ko" : [ "아테네", "테살로니키", "마스트리흐트", "마드리드", "말라가", "맨체스터", "마르세유", "민스크", "모나코", "모스크바", "뮌헨", "무르만스크", "낭트", "나폴리", "나르비크", "니스", "니즈니노브고로드", "노비사드", "뉘른베르크", "라벤나", "레이캬비크", "로마", "상트페테르부르크", "사라고사", "사라예보", "세바스토폴", "세비야", "시에나", "스코페", "소피아", "스플리트", "스트라스부르", "슈투트가르트", "슈체친", "탈린", "탐페레", "헤이그", "토르니오", "트리어", "트리에스테", "아프가니스탄", "알바니아", "알제리", "안도라", "앙골라", "아르헨티나", "아루바", "오스트레일리아", "아제르바이잔", "바하마", "바레인", "방글라데시", "바베이도스", "벨라루스", "벨기에", "벨리즈", "베냉", "버뮤다", "부탄", "볼리비아", "보츠와나", "브라질", "브루나이", "불가리아", "오트볼타", "자메이카", "카자흐스탄", "케니아", "키리바시", "한국", "쿠웨이트", "키르기스스탄", "라오스", "라트비아", "레바논", "레소토", "라이베리아", "리비아", "리히텐슈타인", "리투아니아", "룩셈부르크", "마카오", "마케도니아", "마다가스카르", "말라위", "말레이시아", "몰디브", "말리", "몰타", "마르티니크", "모리타니", "모리셔스", "마요트", "멕시코", "미크로네시아", "몰도바", "모나코", "몽골", "몬테네그로", "몬트세랫", "모로코", "모잠비크", "미얀마", "버마", "나미비아", "나우루", "네팔", "네덜란드", "누벨칼레도니", "뉴칼레도니아", "뉴질랜드", "니카라과", "니제르", "나이지리아", "니우에", "북한", "노르웨이", "오만", "파키스탄", "팔라우", "팔레스타인", "파나마", "파푸아뉴기니", "파라과이", "페루", "필리핀", "폴란드", "포르투갈", "푸에르토리코", "카타르", "레위니옹", "루마니아", "러시아", "르완다", "세인트헬레나", "세인트루시아", "사모아", "산마리노", "사우디아라비아", "세네갈", "세르비아", "세이셸", "시에라리온", "싱가포르", "슬로바키아", "슬로베니아", "소말리아", "스페인", "스리랑카", "수단", "북키프로스", "터키", "튀니지", "트란스니스트리아", "통가", "토켈라우", "토고", "동티모르", "감비아", "탄자니아", "타지키스탄", "시리아", "스위스", "스웨덴", "짐바브웨", "잠비아", "예멘", "서사하라", "웨일스", "베트남", "베네수엘라", "뉴헤브리디스", "바누아투", "우즈베키스탄", "우루과이", "연합왕국", "우크라이나", "우간다", "투발루", "투르크메니스탄"],
    "en" : ["Tirana", "Andorra la Vella", "Yerevan", "Vienna", "Baku", "Minsk", "Brussels", "Sarajevo", "Sofia", "Zagreb", "Nicosia", "Prague", "Copenhagen", "Tallinn", "Helsinki", "Paris", "Tbilisi", "Berlin", "Athens", "Budapest", "Reykjavík", "Dublin", "Rome", "Astana", "Riga", "Vaduz", "Vilnius", "Luxembourg", "Valletta", "Monaco", "Podgorica", "Amsterdam", "Oslo", "Warsaw", "Lisbon", "Bucharest", "Moscow", "San Marino", "Belgrade", "Bratislava", "Ljubljana", "Madrid", "Stockholm", "Bern", "Ankara", "Kiev", "London", "Vatican City", "Jamaica", "Japan", "Kirghizia", "Burma", "Norway", "Portugal", "Poland", "Peru", "Paraguay", "Panama", "Pakistan", "Nigeria", "Niger", "Nicaragua", "Netherlands", "Montenegro", "Mongolia", "Mexico", "Mauritania"],
    "sr" : ["Tirana", "Andora", "Jerevan", "Bec", "Baku", "Minsk", "Brisel", "Sarajevo", "Sofija", "Zagreb", "Nikozija", "Prag", "Kopenhagen", "Talin", "Helsinki", "Pariz", "Tbilisi", "Berlin", "Atina", "Budimpesta", "Rejkjavik", "Dablin", "Rim", "Astana", "Riga", "Vaduc", "Viljnus", "Valeta", "Monako", "Podgorica", "Amsterdam", "Lisabon", "Bukurest", "Moskva", "Beograd", "Bratislava", "Ljubljana", "Madrid", "Stokholm", "Bern", "Ankara", "Kijev", "London", "Vatikan", "Јамајка", "Јапан", "Јордан", "Казахстан", "Кенија", "Кореја", "Кувајт", "Киргизија", "Лаос", "Летонија", "Либан", "Лесото", "Либерија", "Либија", "Лихтенштајн", "Литванија", "Макао", "Македонија", "Мадагаскар", "Малави", "Малезија", "Малдиви", "Мали", "Малта", "Мартиник", "Мауританија", "Маурицијус", "Мајот", "Мексико", "Микронезија", "Молдавија", "Монако", "Монголија", "Монсерат", "Мароко", "Мозамбик", "Мјанмар", "Намибија", "Науру", "Непал", "Холандија", "Никарагва", "Нигер", "Нигерија", "Норвешка", "Оман", "Пакистан", "Палау", "Панама", "Парагвај", "Перу", "Филипини", "Пољска", "Португалија", "Монако", "Порторико", "Катар", "Реинион", "Румунија", "Руанда", "Самоа", "Шкотска", "Сенегал", "Србија", "Сингапур", "Словачка", "Словенија", "Сомалија", "Шпанија", "Судан", "Шведска", "Швајцарска", "Сирија", "Тајван", "Таџикистан", "Танзанија", "Уганда", "Тувалу", "Туркменистан", "Турска", "Тунис", "Тонга", "Токелау", "Того", "Гамбија", "Тајланд", "Уругвај", "Вануату", "Ватикан", "Венецуела", "Вијетнам", "Велс", "Јемен", "Замбија", "Зимбабве"],
    "el" : [ "Λονδίνο", "Παρίσι", "Εδιμβούργο", "Αδριανούπολη", "Αμμόχωστος", "Φλωρεντία", "Φρανκφούρτη", "Φρανκφούρτη", "Γδανσκ", "Γιβραλτάρ", "Γάνδη", "Γλασκώβη", "Γοττίγγη", "Ἐλιβύργη", "Αμβούργο", "Ελσίνκι", "Ηράκλειο", "Άαχεν", "Αλεξανδρούπολη", "Άμστερνταμ", "Αμβέρσα", "Αθήνα", "Άουγκσμπουργκ", "Βαρκελώνη", "Βελιγράδι", "Μπεράτι", "Βερολίνο", "Βέρνη", "Μοναστίρ (Τυνησία)", "Βολωνία", "Μπολτζάνο", "Μπορντώ", "Πρεσβούργο", "Βρυξέλλες", "Βουκουρέστι", "Βουδαπέστη", "Γάδειρα", "Κατάνια", "Κετίγνη", "Χανιά", "Κολωνία", "Κοπεγχάγη", "Κέρκυρα", "Ντέμπρετσεν", "Δρέσδη", "Δυρράχιον", "Μαδρίτη", "Μαγχεστρία", "Μασσαλία", "Ιαμαϊκή", "Ιαπωνία", "Ιορδανία", "Καζακστάν", "Κένυα", "Κιριμπάτι", "Κουβέιτ", "Κιργιζιστάν", "Λάος", "Λάτβια", "Λίβανος", "Λιβερία", "Λιβύη", "Λίχτενσταϊν", "Λιθουανία", "Μακάου", "Μαδαγασκάρη", "Μαλάουι", "Μαλαισία", "Μαλδίβες", "Μαλί", "Μάλτα", "Μαρτινίκα", "Μαυριτανία", "Μαυρίκιος", "Μαγιότ", "Μεξικό", "Μολδαβία", "Μονακό", "Μογγολία", "Μαυροβούνιο", "Μοντσερράτ", "Μαρόκο", "Μοζαμβίκη", "Μυανμάρ", "Βιρμανία", "Ναμίμπια", "Ναουρού", "Νεπάλ", "Ολλανδία", "Νικαράγουα", "Νίγηρας", "Νιγηρία", "Νιούε", "Νορβηγία", "Ομάν", "Πακιστάν", "Παλάου", "Παλαιστίνη", "Παναμάς", "Παραγουάη", "Περού", "Φιλιππίνες", "Πολωνία", "Πορτογαλία", "Κατάρ", "Ρουμανία", "Ρωσία", "Ρουάντα", "Σαμόα", "Σκωτία", "Σερβία", "Σεϋχέλλες", "Σιγκαπούρη", "Σλοβακία", "Σλοβενία", "Σομαλία", "Ισπανία", "Κεϋλάνη", "Σουδάν", "Σουηδία", "Ελβετία", "Συρία", "Ταϊβάν", "Τατζικιστάν", "Τανζανία", "Ταϊλάνδη", "Γκάμπια", "Τόγκο", "Τόγκα", "Τυνησία", "Τουρκία", "Τουρκμενιστάν", "Τουβαλού", "Ουγκάντα"]
}

# Specifies how many words make up one example to be rendered.
words_per_sample = 10

def collect_text_lines(out_dir):
    """
    Collects text in different scripts. Collected text is obtained from Wikipedia. Each language text is palced in
    different file in output directory (file is named according to language code).

    Args:
        out_dir : Output directory path where collected text files are to be placed.
    """
    # Go over all languages
    for language in languages:
        # Set current language.
        wikipedia.set_lang(language)
        search_terms = terms_by_language[language]
        # Open output language file.
        out_lang_file_path = os.path.join(out_dir, language + ".txt")
        out_lang_file = open(out_lang_file_path, "w", encoding="utf-8")
        # Go over all search terms.
        for term in search_terms:
            # Get wiki page summary and split it in words.
            wiki_page = wikipedia.page(term)
            text = wiki_page.summary
            words = text.split()
            # Make text string as composition of certain number of words.
            curr_words = 0
            text_line = ""
            for w in range(len(words)):
                text_line += words[w]
                text_line += " "
                curr_words += 1
                if curr_words == words_per_sample:
                    # We have one line of text, save it to file.
                    out_lang_file.write(text_line)
                    out_lang_file.write("\n")
                    # Reset text line related variables.
                    curr_words = 0
                    text_line = ""

        out_lang_file.close()

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir', help='Path to output dir where samples will be stored.', required=True)
    args = parser.parse_args()
    # Collects text lines and places them to files in output directory.
    collect_text_lines(vars(args)["out_dir"])
