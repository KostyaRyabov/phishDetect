import os
import pandas

headers = {
    'stats': [
        'коэффициент уникальности всех слов',

        #   URL FEATURES

        'наличие ip-адреса в url',
        'использование сервисов сокраения url',

        "наличие сертификата",
        "хороший netloc",

        'длина url',

        'кол-во @ в url',
        'кол-во ! в url',
        'кол-во + в url',
        'кол-во [ и ] в url',
        'кол-во ( и ) в url',
        'кол-во , в url',
        'кол-во $ в url',
        'кол-во ; в url',
        'кол-во пропусков в url',
        'кол-во & в url',
        'кол-во // в url',
        'кол-во / в url',
        'кол-во = в url',
        'кол-во % в url',
        'кол-во ? в url',
        'кол-во _ в url',
        'кол-во - в url',
        'кол-во . в url',
        'кол-во : в url',
        'кол-во * в url',
        'кол-во | в url',
        'кол-во ~ в url',
        'кол-во http токенов в url',

        'https',

        'соотношение цифр в url',
        'кол-во цифр в url',

        "кол-во фишинговых слов в url",
        "кол-во распознанных слов в url",

        'tld в пути url',
        'tld в поддомене url',
        'tld на плохой позиции url',
        'ненормальный поддомен url',

        'кол-во перенаправлений на сайт',
        'кол-во перенаправлений на другие домены',

        'случайный домен',

        'кол-во случайных слов в url',
        'кол-во случайных слов в хосте url',
        'кол-во случайных слов в пути url',

        'кол-во повторяющих последовательностей в url',
        'кол-во повторяющих последовательностей в хосте url',
        'кол-во повторяющих последовательностей в пути url',

        'наличие punycode',
        'домен в брендах',
        'юренд в пути url',
        'кол-во www в url',
        'кол-во com в url',

        'наличие порта в url',

        'кол-во слов в url',
        'средняя длина слова в url',
        'максимальная длина слова в url',
        'минимальная длина слова в url',

        'префикс суффикс в url',

        'кол-во поддоменов',

        'кол-во визульно схожих доменов',

        #   CONTENT FEATURE
        #       (static)

        'степень сжатия страницы',
        'кол-во полей ввода/вывода в основном контексте страницы',
        'соотношение кода в странице в основном контексте страницы',

        'кол-во всех ссылок в основном контексте страницы',
        'соотношение внутренних ссылок на сайты со всеми в основном контексте страницы',
        'соотношение внешних ссылок на сайты со всеми в основном контексте страницы',
        'соотношение пустых ссылок на сайты со всеми в основном контексте страницы',
        "соотношение внутренних CSS со всеми в основном контексте страницы",
        "соотношение внешних CSS со всеми в основном контексте страницы",
        "соотношение встроенных CSS со всеми в основном контексте страницы",
        "соотношение внутренних скриптов со всеми в основном контексте страницы",
        "соотношение внешних скриптов со всеми в основном контексте страницы",
        "соотношение встроенных скриптов со всеми в основном контексте страницы",
        "соотношение внешних изображений со всеми в основном контексте страницы",
        "соотношение внутренних изображений со всеми в основном контексте страницы",
        "общее кол-во перенаправлений по внутренним ссылкам в основном контексте страницы",
        "общее кол-во перенаправлений по внешним ссылкам в основном контексте страницы",
        "общее кол-во ошибок по внутренним ссылкам в основном контексте страницы",
        "общее кол-во ошибок по внешним ссылкам в основном контексте страницы",
        "форма входа в основном контексте страницы",
        "соотношение внешних Favicon со всеми в основном контексте страницы",
        "соотношение внутренних Favicon со всеми в основном контексте страницы",
        "наличие отправки на почту в основном контексте страницы",
        "соотношение внешних медиа со всеми в основном контексте страницы",
        "соотношение внешних медиа со всеми в основном контексте страницы",
        "пустой титульник в основном контексте страницы",
        "соотношение небезопасных якорей со всеми в основном контексте страницы",
        "соотношение безопасных якорей со всеми в основном контексте страницы",
        "соотношение внутренних ссылок на ресурсы со всеми в основном контексте страницы",
        "соотношение внешних ссылок на ресурсы со всеми в основном контексте страницы",
        "наличие невидимых в основном контексте страницы",
        "наличие onmouseover в основном контексте страницы",
        "наличие всплывающих окон в основном контексте страницы",
        "наличие событий правой мыши в основном контексте страницы",
        "наличие домена в тексте в основном контексте страницы",
        "наличие домена в титульнике в основном контексте страницы",
        "домен с авторскими правами в основном контексте страницы",
        "кол-во фишинговых слов в тексте в основном контексте страницы",
        "кол-во слов в тексте в основном контексте страницы",
        "соотношение текста со всех изображений с основным текстом в основном контексте страницы",
        "соотношение текста внутренних изображений с основным текстом в основном контексте страницы",
        "соотношение текста внешних изображений с основным текстом в основном контексте страницы",
        "соотношение текста внешних изображений с текстом внутренних изображений в основном контексте страницы",

        #       (dynamic)

        "соотношение основного текста с динамически добавляемым текстом страницы",

        #       (dynamic internals)

        "соотношение основного текста с внутреннее добавляемым текстом страницы",
        "соотношение кода в внутренне добавляемом контексте страницы",
        "кол-во полей ввода/вывода в внутренне добавляемом контексте страницы",

        'кол-во всех ссылок во внутренне добавляемом контексте страницы',
        'соотношение внутренних ссылок на сайты со всеми во внутренне добавляемом контексте страницы',
        'соотношение внешних ссылок на сайты со всеми во внутренне добавляемом контексте страницы',
        'соотношение пустых ссылок на сайты со всеми во внутренне добавляемом контексте страницы',
        "соотношение внутренних CSS со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних CSS со всеми во внутренне добавляемом контексте страницы",
        "соотношение встроенных CSS со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних скриптов со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних скриптов со всеми во внутренне добавляемом контексте страницы",
        "соотношение встроенных скриптов со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних изображений со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних изображений со всеми во внутренне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внутренним ссылкам во внутренне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внешним ссылкам во внутренне добавляемом контексте страницы",
        "общее кол-во ошибок по внутренним ссылкам во внутренне добавляемом контексте страницы",
        "общее кол-во ошибок по внешним ссылкам во внутренне добавляемом контексте страницы",
        "форма входа во внутренне добавляемом контексте страницы",
        "соотношение внешних Favicon со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних Favicon со всеми во внутренне добавляемом контексте страницы",
        "наличие отправки на почту во внутренне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внутренне добавляемом контексте страницы",
        "пустой титульник во внутренне добавляемом контексте страницы",
        "соотношение небезопасных якорей со всеми во внутренне добавляемом контексте страницы",
        "соотношение безопасных якорей со всеми во внутренне добавляемом контексте страницы",
        "соотношение внутренних ссылок на ресурсы со всеми во внутренне добавляемом контексте страницы",
        "соотношение внешних ссылок на ресурсы со всеми во внутренне добавляемом контексте страницы",
        "наличие невидимых во внутренне добавляемом контексте страницы",
        "наличие onmouseover во внутренне добавляемом контексте страницы",
        "наличие всплывающих окон во внутренне добавляемом контексте страницы",
        "наличие событий правой мыши во внутренне добавляемом контексте страницы",
        "наличие домена в тексте во внутренне добавляемом контексте страницы",
        "наличие домена в титульнике во внутренне добавляемом контексте страницы",
        "домен с авторскими правами во внутренне добавляемом контексте страницы",

        "кол-во операций ввода/вывода во внутренне добавляемом коде страницы",
        "кол-во фишинговых слов во внутренне добавляемом контексте страницы",
        "кол-во слов во внутренне добавляемом контексте страницы",

        #       (dynamic externals)

        "соотношение основного текста с внешне добавляемым текстом страницы",
        "соотношение кода в внешне добавляемом контексте страницы",
        "кол-во полей ввода/вывода в внешне добавляемом контексте страницы",

        'кол-во всех ссылок во внешне добавляемом контексте страницы',
        'соотношение внутренних ссылок на сайты со всеми во внешне добавляемом контексте страницы',
        'соотношение внешних ссылок на сайты со всеми во внешне добавляемом контексте страницы',
        'соотношение пустых ссылок на сайты со всеми во внешне добавляемом контексте страницы',
        "соотношение внутренних CSS со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних CSS со всеми во внешне добавляемом контексте страницы",
        "соотношение встроенных CSS со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних скриптов со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних скриптов со всеми во внешне добавляемом контексте страницы",
        "соотношение встроенных скриптов со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних изображений со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних изображений со всеми во внешне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внутренним ссылкам во внешне добавляемом контексте страницы",
        "общее кол-во перенаправлений по внешним ссылкам во внешне добавляемом контексте страницы",
        "общее кол-во ошибок по внутренним ссылкам во внешне добавляемом контексте страницы",
        "общее кол-во ошибок по внешним ссылкам во внешне добавляемом контексте страницы",
        "форма входа во внешне добавляемом контексте страницы",
        "соотношение внешних Favicon со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних Favicon со всеми во внешне добавляемом контексте страницы",
        "наличие отправки на почту во внешне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних медиа со всеми во внешне добавляемом контексте страницы",
        "пустой титульник во внешне добавляемом контексте страницы",
        "соотношение небезопасных якорей со всеми во внешне добавляемом контексте страницы",
        "соотношение безопасных якорей со всеми во внешне добавляемом контексте страницы",
        "соотношение внутренних ссылок на ресурсы со всеми во внешне добавляемом контексте страницы",
        "соотношение внешних ссылок на ресурсы со всеми во внешне добавляемом контексте страницы",
        "наличие невидимых во внешне добавляемом контексте страницы",
        "наличие onmouseover во внешне добавляемом контексте страницы",
        "наличие всплывающих окон во внешне добавляемом контексте страницы",
        "наличие событий правой мыши во внешне добавляемом контексте страницы",
        "наличие домена в тексте во внешне добавляемом контексте страницы",
        "наличие домена в титульнике во внешне добавляемом контексте страницы",
        "домен с авторскими правами во внешне добавляемом контексте страницы",

        "кол-во операций ввода/вывода во внешне добавляемом коде страницы",
        "кол-во фишинговых слов во внешне добавляемом контексте страницы",
        "кол-во слов во внешне добавляемом контексте страницы",

        #   EXTERNAL FEATURES

        'срок регистрации домена',
        "домен зарегестрирован",
        "рейтинг по Alexa",
        "рейтинг по openpagerank",
        "соотношение оставшегося времени действия сертификата",
        "срок действия сертификата",
        "кол-во альтернативных имен в сертификате"
    ],
    # 'stats': [
    #     'word_ratio',
    #
    #                 #   URL FEATURES
    #
    #                 'uf.having_ip_address(url)',
    #                 'uf.shortening_service(url)',
    #
    #                 "cert!=None",
    #                 "good_netloc",
    #
    #                 'uf.url_length(r_url)',
    #
    #                 'uf.count_at(r_url)',
    #                 'uf.count_exclamation(r_url)',
    #                 'uf.count_plust(r_url)',
    #                 'uf.count_sBrackets(r_url)',
    #                 'uf.count_rBrackets(r_url)',
    #                 'uf.count_comma(r_url)',
    #                 'uf.count_dollar(r_url)',
    #                 'uf.count_semicolumn(r_url)',
    #                 'uf.count_space(r_url)',
    #                 'uf.count_and(r_url)',
    #                 'uf.count_double_slash(r_url)',
    #                 'uf.count_slash(r_url)',
    #                 'uf.count_equal(r_url)',
    #                 'uf.count_percentage(r_url)',
    #                 'uf.count_question(r_url)',
    #                 'uf.count_underscore(r_url)',
    #                 'uf.count_hyphens(r_url)',
    #                 'uf.count_dots(r_url)',
    #                 'uf.count_colon(r_url)',
    #                 'uf.count_star(r_url)',
    #                 'uf.count_or(r_url)',
    #                 'uf.count_tilde(r_url)',
    #                 'uf.count_http_token(r_url)',
    #
    #                 'uf.https_token(scheme)',
    #
    #                 'uf.ratio_digits(r_url)',
    #                 'uf.count_digits(r_url)',
    #
    #                 "cf.count_phish_hints(url_words, phish_hints)",
    #                 "(len(url_words), 0)",
    #
    #                 'uf.tld_in_path(tld,path)',
    #                 'uf.tld_in_subdomain(tld,subdomain)',
    #                 'uf.tld_in_bad_position(tld,subdomain,path)',
    #                 'uf.abnormal_subdomain(r_url)',
    #
    #                 'uf.count_redirection(request)',
    #                 'uf.count_external_redirection(request,domain)',
    #
    #                 'uf.random_domain(second_level_domain)',
    #
    #                 'uf.random_words(words_raw)',
    #                 'uf.random_words(words_raw_host)',
    #                 'uf.random_words(words_raw_path)',
    #
    #                 'uf.char_repeat(words_raw)',
    #                 'uf.char_repeat(words_raw_host)',
    #                 'uf.char_repeat(words_raw_path)',
    #
    #                 'uf.punycode(r_url)',
    #                 'uf.domain_in_brand(second_level_domain)',
    #                 'uf.brand_in_path(second_level_domain, words_raw_path)',
    #                 'uf.check_www(words_raw)',
    #                 'uf.check_com(words_raw)',
    #
    #                 'uf.port(r_url)',
    #
    #                 'uf.length_word_raw(words_raw)',
    #                 'uf.average_word_length(words_raw)',
    #                 'uf.longest_word_length(words_raw)',
    #                 'uf.shortest_word_length(words_raw)',
    #
    #                 'uf.prefix_suffix(r_url)',
    #
    #                 'uf.count_subdomain(r_url)',
    #
    #                 'uf.count_visual_similarity_domains(second_level_domain)',
    #
    #                 #   CONTENT FEATURE
    #                 #       (static)
    #
    #                 'cf.compression_ratio(request)',
    #                 'cf.count_textareas(content)',
    #                 'cf.ratio_js_on_html(Text)',
    #
    #                 'len(iUrl_s)+len(eUrl_s)',
    #                 'cf.urls_ratio(iUrl_s,iUrl_s+eUrl_s+nUrl_s)',
    #                 'cf.urls_ratio(eUrl_s,iUrl_s+eUrl_s+nUrl_s)',
    #                 'cf.urls_ratio(nUrl_s,iUrl_s+eUrl_s+nUrl_s)',
    #                 "cf.ratio_List(CSS,'internals')",
    #                 "cf.ratio_List(CSS,'externals')",
    #                 "cf.ratio_List(CSS,'embedded')",
    #                 "cf.ratio_List(SCRIPT,'internals')",
    #                 "cf.ratio_List(SCRIPT,'externals')",
    #                 "cf.ratio_List(SCRIPT,'embedded')",
    #                 "cf.ratio_List(Img,'externals')",
    #                 "cf.ratio_List(Img,'internals')",
    #                 "cf.count_reqs_redirections(reqs_iData_s)",
    #                 "cf.count_reqs_redirections(reqs_eData_s)",
    #                 "cf.count_reqs_error(reqs_iData_s)",
    #                 "cf.count_reqs_error(reqs_eData_s)",
    #                 "cf.login_form(Form)",
    #                 "cf.ratio_List(Favicon,'externals')",
    #                 "cf.ratio_List(Favicon,'internals')",
    #                 "cf.submitting_to_email(Form)",
    #                 "cf.ratio_List(Media,'internals')",
    #                 "cf.ratio_List(Media,'externals')",
    #                 "cf.empty_title(Title)",
    #                 "cf.ratio_anchor(Anchor,'unsafe')",
    #                 "cf.ratio_anchor(Anchor,'safe')",
    #                 "cf.ratio_List(Link,'internals')",
    #                 "cf.ratio_List(Link,'externals')",
    #                 "cf.iframe(IFrame)",
    #                 "cf.onmouseover(content)",
    #                 "cf.popup_window(content)",
    #                 "cf.right_clic(content)",
    #                 "cf.domain_in_text(second_level_domain,Text)",
    #                 "cf.domain_in_text(second_level_domain,Title)",
    #                 "cf.domain_with_copyright(domain,content)",
    #                 "cf.count_phish_hints(Text, phish_hints)",
    #                 "(len(sContent_words), 0)",
    #                 "cf.ratio_Txt(iImgTxt_words+eImgTxt_words,sContent_words)",
    #                 "cf.ratio_Txt(iImgTxt_words,sContent_words)",
    #                 "cf.ratio_Txt(eImgTxt_words,sContent_words)",
    #                 "cf.ratio_Txt(eImgTxt_words,iImgTxt_words)",
    #
    #                 #       (dynamic)
    #
    #                 "cf.ratio_dynamic_html(Text,"".join([Text_di,Text_de]))",
    #
    #                 #       (dynamic internals)
    #
    #                 "cf.ratio_dynamic_html(Text,Text_di)",
    #                 "cf.ratio_js_on_html(Text_di)",
    #                 "cf.count_textareas(content_di)",
    #
    #                 "len(iUrl_di)+len(eUrl_di)",
    #                 "cf.urls_ratio(iUrl_di,iUrl_di+eUrl_di+nUrl_di)",
    #                 "cf.urls_ratio(eUrl_di,iUrl_di+eUrl_di+nUrl_di)",
    #                 "cf.urls_ratio(nUrl_di,iUrl_di+eUrl_di+nUrl_di)",
    #                 "cf.ratio_List(CSS_di,'internals')",
    #                 "cf.ratio_List(CSS_di,'externals')",
    #                 "cf.ratio_List(CSS_di,'embedded')",
    #                 "cf.ratio_List(SCRIPT_di,'internals')",
    #                 "cf.ratio_List(SCRIPT_di,'externals')",
    #                 "cf.ratio_List(SCRIPT_di,'embedded')",
    #                 "cf.ratio_List(Img_di,'externals')",
    #                 "cf.ratio_List(Img_di,'internals')",
    #                 "cf.count_reqs_redirections(reqs_iData_di)",
    #                 "cf.count_reqs_redirections(reqs_eData_di)",
    #                 "cf.count_reqs_error(reqs_iData_di)",
    #                 "cf.count_reqs_error(reqs_eData_di)",
    #                 "cf.login_form(Form_di)",
    #                 "cf.ratio_List(Favicon_di,'externals')",
    #                 "cf.ratio_List(Favicon_di,'internals')",
    #                 "cf.submitting_to_email(Form_di)",
    #                 "cf.ratio_List(Media_di,'internals')",
    #                 "cf.ratio_List(Media_di,'externals')",
    #                 "cf.empty_title(Title_di)",
    #                 "cf.ratio_anchor(Anchor_di,'unsafe')",
    #                 "cf.ratio_anchor(Anchor_di,'safe')",
    #                 "cf.ratio_List(Link_di,'internals')",
    #                 "cf.ratio_List(Link_di,'externals')",
    #                 "cf.iframe(IFrame_di)",
    #                 "cf.onmouseover(content_di)",
    #                 "cf.popup_window(content_di)",
    #                 "cf.right_clic(content_di)",
    #                 "cf.domain_in_text(second_level_domain,Text_di)",
    #                 "cf.domain_in_text(second_level_domain,Title_di)",
    #                 "cf.domain_with_copyright(domain,content_di)",
    #
    #                 "cf.count_io_commands(internals_script_doc)",
    #                 "cf.count_phish_hints(Text_di,phish_hints)",
    #                 "(len(diContent_words), 0)",
    #
    #                 #       (dynamic externals)
    #
    #                 "cf.ratio_dynamic_html(Text,Text_de)",
    #                 "cf.ratio_js_on_html(Text_de)",
    #                 "cf.count_textareas(content_de)",
    #
    #                 "len(iUrl_de)+len(eUrl_de)",
    #                 "cf.urls_ratio(iUrl_de,iUrl_de+eUrl_de+nUrl_de)",
    #                 "cf.urls_ratio(eUrl_de,iUrl_de+eUrl_de+nUrl_de)",
    #                 "cf.urls_ratio(nUrl_de,iUrl_de+eUrl_de+nUrl_de)",
    #                 "cf.ratio_List(CSS_de,'internals')",
    #                 "cf.ratio_List(CSS_de,'externals')",
    #                 "cf.ratio_List(CSS_de,'embedded')",
    #                 "cf.ratio_List(SCRIPT_de,'internals')",
    #                 "cf.ratio_List(SCRIPT_de,'externals')",
    #                 "cf.ratio_List(SCRIPT_de,'embedded')",
    #                 "cf.ratio_List(Img_de,'externals')",
    #                 "cf.ratio_List(Img_de,'internals')",
    #                 "cf.count_reqs_redirections(reqs_iData_de)",
    #                 "cf.count_reqs_redirections(reqs_eData_de)",
    #                 "cf.count_reqs_error(reqs_iData_de)",
    #                 "cf.count_reqs_error(reqs_eData_de)",
    #                 "cf.login_form(Form_de)",
    #                 "cf.ratio_List(Favicon_de,'externals')",
    #                 "cf.ratio_List(Favicon_de,'internals')",
    #                 "cf.submitting_to_email(Form_de)",
    #                 "cf.ratio_List(Media_de,'internals')",
    #                 "cf.ratio_List(Media_de,'externals')",
    #                 "cf.empty_title(Title_de)",
    #                 "cf.ratio_anchor(Anchor_de,'unsafe')",
    #                 "cf.ratio_anchor(Anchor_de,'safe')",
    #                 "cf.ratio_List(Link_de,'internals')",
    #                 "cf.ratio_List(Link_de,'externals')",
    #                 "cf.iframe(IFrame_de)",
    #                 "cf.onmouseover(content_de)",
    #                 'cf.popup_window(content_de)',
    #                 "cf.right_clic(content_de)",
    #                 "cf.domain_in_text(second_level_domain,Text_de)",
    #                 "cf.domain_in_text(second_level_domain,Title_de)",
    #                 "cf.domain_with_copyright(domain,content_de)",
    #
    #                 "cf.count_io_commands(externals_script_doc)",
    #                 "cf.count_phish_hints(Text_de,phish_hints)",
    #                 "(len(deContent_words), 0)",
    #
    #                 #   EXTERNAL FEATURES
    #
    #                 'ef.domain_registration_length(domain)',
    #                 "ef.whois_registered_domain(domain)",
    #                 "ef.web_traffic(r_url)",
    #                 "ef.page_rank(domain)",
    #                 "ef.remainder_valid_cert(hostinfo.cert)",
    #                 "ef.valid_cert_period(hostinfo.cert)",
    #                 "ef.count_alt_names(hostinfo.cert)"
    # ],
    'metadata': [
        'url',
        'lang',
        'status'
    ],
    'substats': [
        'extraction-contextData-time',
        'image-recognition-time'
    ]
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import json
import pickle
from shutil import move

import telepot

telegram_info = pandas.read_csv('telegram_client.csv')
bot = telepot.Bot(telegram_info['BOT_token'][0])

def handle(msg):
    chat_id = msg['chat']['id']
    # command = msg['text']

    try:
        with open("data/trials/metric.txt") as f:
            bot.sendMessage(chat_id, float(f.read().strip()))
        for metric in ['accuracy', 'loss']:
            bot.sendPhoto(chat_id, photo=open('data/trials/{}.png'.format(metric), 'rb'))
        with open("data/trials/space.json") as f:
            bot.sendMessage(chat_id, str(f.read()))
    except:
        bot.sendMessage(chat_id, 'error')

bot.message_loop(handle)

tf.compat.v1.enable_eager_execution()


def NN():
    frame = pandas.read_csv('data/datasets/OUTPUT/dataset.csv')

    cols = [col for col in headers['stats'] if col in list(frame)]

    X = frame[cols].to_numpy()
    Y = frame['status'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    try:
        with open("data/trials/results.pkl", 'rb') as file:
            trials = pickle.load(file)
    except:
        trials = Trials()

    def layer(N, M=-1):
        if M==-1:
            M = N
        if N == 0:
            return hp.choice('layer_{}'.format(M-N),[None])
        return hp.choice('layer_{}'.format(M-N),[
            None,
            {
                'activation': hp.choice('activation_{}'.format(M - N), [
                    None,
                    'selu',
                    'relu',
                    'softmax',
                    'sigmoid',
                    'softplus',
                    'softsign',
                    'tanh',
                    'elu',
                    'exponential'
                ]),
                'nodes_count': hp.randint('nodes_count_{}'.format(M - N), 400) + 2,
                'dropout': hp.uniform('dropout_{}'.format(M - N), 0, 0.8),
                'BatchNormalization': hp.choice('BatchNormalization_{}'.format(M - N), [False, True]),
                'next': layer(N - 1, M)
            }
        ])

    space = {
        'decay_steps': hp.choice('decay_steps', [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]),
        'layers': layer(4),
        'optimizer':
        hp.choice('optimizer', [
            {
                'type': 'Adadelta',
                'learning_rate': hp.uniform('Adadelta_lr', 0.001, 1),
            },
            {
                'type': 'Adagrad',
                'learning_rate': hp.uniform('Adagrad_lr', 0.001, 1),
            },
            {
                'type': 'Adamax',
                'learning_rate': hp.uniform('Adamax_lr', 0.001, 1),
            },
            {
                'type': 'Adam',
                'learning_rate': hp.uniform('Adam_lr', 0.001, 1),
                'amsgrad': hp.choice('Adam_amsgrad', [False, True])
            },
            {
                'type': 'Ftrl',
                'learning_rate': hp.uniform('Ftrl_lr', 0.001, 1),
            },
            {
                'type': 'Nadam',
                'learning_rate': hp.uniform('Nadam_lr', 0.001, 1),
            },
            {
                'type': 'RMSprop',
                'learning_rate': hp.uniform('RMSprop_lr', 0.001, 1),
                'centered': hp.choice('RMSprop_centered', [False, True]),
                'momentum': hp.uniform('RMSprop_momentum', 0.001, 1),
            },
            {
                'type': 'SGD',
                'learning_rate': hp.uniform('SGD_lr', 0.001, 1),
                'nesterov': hp.choice('SGD_nesterov', [False,True]),
                'momentum': hp.uniform('SGD_momentum', 0.001, 1),
            }
         ]),
        'batch_size': hp.choice('batch_size', [None, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
        'init': hp.choice('init', [
                'glorot_normal',
                'truncated_normal',
                'glorot_uniform'
            ]),
        'trainable_BatchNormalization': hp.choice('trainable_BatchNormalization', [False,True]),
        'trainable_dropouts': hp.choice('trainable_dropouts', [False,True]),
        'shuffle': hp.choice('shuffle', [False,True])
    }

    def objective(space):
        model = models.Sequential()

        layer = space['layers']

        while layer:
            model.add(layers.Dense(
                layer['nodes_count'],
                kernel_initializer=space['init'],
                activation=layer['activation'])
            )
            if layer['dropout'] >= 0.01:
                model.add(layers.Dropout(layer['dropout'], trainable=space['trainable_dropouts']))
            if layer['BatchNormalization']:
                model.add(layers.BatchNormalization(trainable=space['trainable_BatchNormalization']))

            layer = layer['next']

        model.add(layers.Dense(1, kernel_initializer=space['init'], activation='sigmoid'))

        def scheduler(epoch, lr):
            return lr * tf.math.exp(-epoch/space['decay_steps'])

        def tf_callbacks():
            return [
                tf.keras.callbacks.ModelCheckpoint(
                    'data/models/NN/tmp.h5',
                    monitor='accuracy',
                    mode='max',
                    verbose=0,
                    save_best_only=True
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=50,
                    # min_delta=0.000001,
                    mode='max',
                    verbose=0),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=50,
                    # min_delta=0.0001,
                    mode='min',
                    verbose=0),
                tf.keras.callbacks.LearningRateScheduler(scheduler)
            ]

        if space['optimizer']['type'] == 'Adadelta':
            optimizer = optimizers.Adadelta(learning_rate=space['optimizer']['learning_rate'])
        elif space['optimizer']['type'] == 'Adagrad':
            optimizer = optimizers.Adagrad(learning_rate=space['optimizer']['learning_rate'],)
        elif space['optimizer']['type'] == 'Adam':
            optimizer = optimizers.Adam(learning_rate=space['optimizer']['learning_rate'],
                                        amsgrad=space['optimizer']['amsgrad'])
        elif space['optimizer']['type'] == 'Adamax':
            optimizer = optimizers.Adamax(learning_rate=space['optimizer']['learning_rate'])
        elif space['optimizer']['type'] == 'Ftrl':
            optimizer = optimizers.Ftrl(learning_rate=space['optimizer']['learning_rate'])
        elif space['optimizer']['type'] == 'Nadam':
            optimizer = optimizers.Nadam(learning_rate=space['optimizer']['learning_rate'])
        elif space['optimizer']['type'] == 'RMSprop':
            optimizer = optimizers.RMSprop(learning_rate=space['optimizer']['learning_rate'],
                                           centered=space['optimizer']['centered'],
                                           momentum=space['optimizer']['momentum'])
        elif space['optimizer']['type'] == 'SGD':
            optimizer = optimizers.SGD(learning_rate=space['optimizer']['learning_rate'],
                                       nesterov=space['optimizer']['nesterov'],
                                       momentum=space['optimizer']['momentum'])


        model.compile(
            optimizer=optimizer,
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        history = model.fit(
            x_train, y_train,
            validation_split=0.3,
            epochs=2000,
            callbacks=tf_callbacks(),
            verbose=0,
            batch_size=space['batch_size'],
            shuffle=space['shuffle']
        )

        # y_pred = model.predict(x_test, verbose=0)
        # try:
        #     score = roc_auc_score(y_test, y_pred)
        #     loss, acc = model.evaluate(x_test, y_test, verbose=0)
        #     return {'loss': -score, 'status': STATUS_OK, 'score': acc, 'loss_': loss, 'epochs': len(history.history['loss'])}
        # except:
        #     return {'loss': 0, 'status': STATUS_OK, 'score': 0, 'loss_': 0, 'epochs': 0}

        loss, acc = model.evaluate(x_test, y_test, verbose=0)

        try:
            with open("data/trials/metric.txt") as f:
                max_acc = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            max_acc = -1

        if acc > max_acc:
            model.save("data/models/NN/nn1.h5")
            move('data/models/NN/tmp.h5', "data/models/NN/nn2.h5")
            with open("data/trials/space.json", "w") as f:
                f.write(str(space))
            with open("data/trials/metric.txt", "w") as f:
                f.write(str(acc))
            with open("data/trials/history.pkl", 'wb') as f:
                pickle.dump(history.history, f)

            for metric in ['accuracy', 'loss']:
                draw(history.history, metric)

        return {'loss': -acc, 'status': STATUS_OK, 'history': history.history, 'space': space}

    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=2000+len(trials),
        trials=trials,
        timeout=60*60*10
    )

    def typer(o):
        if isinstance(o, np.int32): return int(o)
        return o

    with open("data/trials/best.json", "w") as f:
        json.dump(best, f, default=typer)

    with open("data/trials/results.pkl", 'wb') as output:
        pickle.dump(trials, output)


def draw(history, metric):
    plt.plot(history[metric])
    plt.plot(history['val_{}'.format(metric)])
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('data/trials/{}.png'.format(metric))
    plt.close()