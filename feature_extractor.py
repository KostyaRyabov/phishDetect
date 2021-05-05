from main import state

if state in range(2, 10) or state == 48:
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
            'бренд в пути url',
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
            "соотношение внутренних медиа со всеми в основном контексте страницы",
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
            "соотношение внутренних медиа со всеми во внутренне добавляемом контексте страницы",
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
            "соотношение внутренних медиа со всеми во внешне добавляемом контексте страницы",
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


if state in range(1, 7):
    import re
    import requests
    import tldextract
    import concurrent.futures
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, urlsplit, urljoin
    from tools import tokenize, segment, clear_text, benchmark

    @benchmark(10)
    def is_URL_accessible(url, time_out=5):
        page = None
        try:
            page = requests.get(url, timeout=time_out)
        except:
            parsed = urlparse(url)
            url = parsed.scheme + '://' + parsed.netloc
            if not parsed.netloc.startswith('www'):
                url = parsed.scheme + '://www.' + parsed.netloc
                try:
                    page = requests.get(url, timeout=time_out)
                except:
                    page = None
                    pass

        if page and page.status_code == 200 and page.content not in ["b''", "b' '"]:
            return True, page
        else:
            return False, None


if state in range(2, 7):
    import time
    import asyncio
    import threading
    import numpy as np
    import url_features as uf
    from tqdm import tqdm
    import content_features as cf
    import external_features as ef
    from googletrans import Translator
    from data.collector import dir_path
    from tools import compute_tf

    key = open("OPR_key.txt").read()
    translator = Translator()

    def load_phishHints():
        hints_dir = "data/phish_hints/"
        file_list = os.listdir(hints_dir)

        if file_list:
            return {leng[0:2]: pandas.read_csv(hints_dir + leng, header=None)[0].tolist() for leng in file_list}
        else:
            hints = {'en': [
                'login',
                'logon',
                'sign',
                'account',
                'authorization',
                'registration',
                'user',
                'password',
                'pay',
                'name',
                'profile',
                'mail',
                'pass',
                'reg',
                'log',
                'auth',
                'psw',
                'nickname',
                'enter',
                'bank',
                'card',
                'pincode',
                'phone',
                'key',
                'visa',
                'cvv',
                'cvp',
                'cvc',
                'ccv'
            ]
            }

            data = pandas.DataFrame(hints)
            filename = "data/phish_hints/en.csv"
            data.to_csv(filename, index=False, header=False)

            return hints

    phish_hints = load_phishHints()


    def check_Language(text):
        global phish_hints

        language = translator.detect(str(text)[0:5000]).lang

        if language not in phish_hints.keys():
            words = translator.translate(" ".join(phish_hints['en'][:25]), src='en', dest=language).text.split(" ")

            phish_hints[language] = [str(word.strip()) for word in words]

            data = pandas.DataFrame(phish_hints[language])
            filename = "data/phish_hints/{0}.csv".format(language)
            data.to_csv(filename, index=False, header=False)

        return language


    def get_domain(url):
        o = urlsplit(url)
        return o.hostname, tldextract.extract(url).domain, o.path, o.netloc


    @benchmark(100)
    def extract_data_from_URL(hostname, content, domain, base_url):
        Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                       "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

        Href = {'internals': [], 'externals': [], 'null': []}
        Link = {'internals': [], 'externals': [], 'null': []}
        Anchor = {'safe': [], 'unsafe': [], 'null': []}
        Img = {'internals': [], 'externals': [], 'null': []}
        Media = {'internals': [], 'externals': [], 'null': []}
        Form = {'internals': [], 'externals': [], 'null': []}
        CSS = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}
        Favicon = {'internals': [], 'externals': [], 'null': []}
        IFrame = {'visible': [], 'invisible': [], 'null': []}
        SCRIPT = {'internals': [], 'externals': [], 'null': [], 'embedded': 0}  # JavaScript
        Title = ''
        Text = ''

        soup = BeautifulSoup(content, 'html.parser')    # lxml

        # collect all external and internal hrefs from url
        for script in soup.find_all('script', src=True):
            url = script['src']

            if url in Null_format:
                url = 'http://' + hostname + '/' + url
                SCRIPT['null'].append(url)
                Link['null'].append(url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                SCRIPT['internals'].append(url)
                Link['internals'].append(url)
            else:
                SCRIPT['externals'].append(url)
                Link['externals'].append(url)

        # collect all external and internal hrefs from url
        for href in soup.find_all('a', href=True):
            url = href['href']

            if "#" in url or "javascript" in url.lower() or "mailto" in url.lower():
                Anchor['unsafe'].append('http://' + hostname + '/' + url)

            if url in Null_format:
                Href['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Href['internals'].append(url)
            else:
                Href['externals'].append(url)
                Anchor['safe'].append(url)

        # collect all media src tags
        for img in soup.find_all('img', src=True):
            url = img['src']

            if url in Null_format:
                url = 'http://' + hostname + '/' + url
                Media['null'].append(url)
                Img['null'].append(url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
                Img['internals'].append(url)
            else:
                Media['externals'].append(url)
                Img['externals'].append(url)

        for audio in soup.find_all('audio', src=True):
            url = audio['src']

            if url in Null_format:
                Media['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
            else:
                Media['externals'].append(url)

        for embed in soup.find_all('embed', src=True):
            url = embed['src']

            if url in Null_format:
                Media['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
            else:
                Media['externals'].append(url)

        for i_frame in soup.find_all('iframe', src=True):
            url = i_frame['src']

            if url in Null_format:
                Media['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Media['internals'].append(url)
            else:
                Media['externals'].append(url)

        # collect all link tags
        for link in soup.findAll('link', href=True):
            url = link['href']

            if url in Null_format:
                Link['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Link['internals'].append(url)
            else:
                Link['externals'].append(url)

        # collect all css
        for link in soup.find_all('link', rel='stylesheet'):
            url = link['href']

            if url in Null_format:
                CSS['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                CSS['internals'].append(url)
            else:
                CSS['externals'].append(url)

        CSS['embedded'] = len([css for css in soup.find_all('style', type='text/css') if len(css.contents) > 0])

        # collect all form actions
        for form in soup.findAll('form', action=True):
            url = form['action']

            if url in Null_format or url == 'about:best_nn':
                Form['null'].append('http://' + hostname + '/' + url)
                continue

            url = urljoin(base_url, url)

            if domain in urlparse(url).netloc:
                Form['internals'].append(url)
            else:
                Form['externals'].append(url)

        # collect all link tags
        for head in soup.find_all('head'):
            for head.link in soup.find_all('link', href=True):
                url = head.link['href']

                if url in Null_format:
                    Favicon['null'].append('http://' + hostname + '/' + url)
                    continue

                url = urljoin(base_url, url)

                if domain in urlparse(url).netloc:
                    Favicon['internals'].append(url)
                else:
                    Favicon['externals'].append(url)

            for head.link in soup.findAll('link', {'href': True, 'rel': True}):
                isicon = False
                if isinstance(head.link['rel'], list):
                    for e_rel in head.link['rel']:
                        if e_rel.endswith('icon'):
                            isicon = True
                            break
                else:
                    if head.link['rel'].endswith('icon'):
                        isicon = True
                        break

                if isicon:
                    url = head.link['href']

                    if url in Null_format:
                        Favicon['null'].append('http://' + hostname + '/' + url)
                        continue

                    url = urljoin(base_url, url)

                    if domain in urlparse(url).netloc:
                        Favicon['internals'].append(url)
                    else:
                        Favicon['externals'].append(url)

        # collect i_frame
        for i_frame in soup.find_all('iframe', width=True, height=True, frameborder=True):
            if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['frameborder'] == "0":
                IFrame['invisible'].append(i_frame)
            else:
                IFrame['visible'].append(i_frame)
        for i_frame in soup.find_all('iframe', width=True, height=True, border=True):
            if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['border'] == "0":
                IFrame['invisible'].append(i_frame)
            else:
                IFrame['visible'].append(i_frame)
        for i_frame in soup.find_all('iframe', width=True, height=True, style=True):
            if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['style'] == "border:none;":
                IFrame['invisible'].append(i_frame)
            else:
                IFrame['visible'].append(i_frame)

        # get page title
        try:
            Title = soup.title.string.lower()
        except:
            pass

        # get content text
        Text = soup.get_text().lower()

        def merge_scripts(script_lnks):
            docs = []

            for url in script_lnks:
                state, request = is_URL_accessible(url)[0]

                if state:
                    docs.append(str(request.content))

            return "\n".join(docs)

        internals_script_doc = merge_scripts(SCRIPT['internals'])
        externals_script_doc = merge_scripts(SCRIPT['externals'])

        try:
            internals_script_doc = ' '.join(
                [internals_script_doc] + [script.contents[0] for script in soup.find_all('script', src=False) if
                                          len(script.contents) > 0])
            SCRIPT['embedded'] = len(
                [script.contents[0] for script in soup.find_all('script', src=False) if len(script.contents) > 0])
        except:
            pass

        return Href, Link, Anchor, Media, Img, Form, CSS, Favicon, IFrame, SCRIPT, Title, Text, internals_script_doc, externals_script_doc


    import configparser

    config = configparser.ConfigParser()
    config.read('settings.ini')
    http_request_group_thread_count = int(config['THREADS']['http_request_group_thread_count'])
    extractor_thread_count = int(config['THREADS']['extractor_thread_count'])
    cpu_base_tasks_thread_count = int(config['THREADS']['cpu_base_tasks_thread_count'])
    image_recognition_thread_count = int(config['THREADS']['image_recognition_thread_count'])

    requests_sem = threading.Semaphore(http_request_group_thread_count)
    OCR_sem = threading.Semaphore(image_recognition_thread_count)
    run_sem = threading.Semaphore(cpu_base_tasks_thread_count)


    def extract_features(url, status):
        def words_raw_extraction(domain, subdomain, path):
            w_domain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
            w_subdomain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())
            w_path = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
            raw_words = w_domain + w_path + w_subdomain
            w_host = w_domain + w_subdomain
            return segment(list(filter(None, raw_words))), \
                   segment(list(filter(None, w_host))), \
                   segment(list(filter(None, w_path)))

        with requests_sem:
            resolve = is_URL_accessible(url)[0]
        if type(resolve) == tuple:
            (state, request) = resolve
        else:
            return None

        if state:
            r_url = request.url
            content = str(request.content)
            hostname, second_level_domain, path, netloc = get_domain(r_url)
            extracted_domain = tldextract.extract(r_url)
            domain = extracted_domain.domain + '.' + extracted_domain.suffix
            subdomain = extracted_domain.subdomain
            tmp = r_url[r_url.find(extracted_domain.suffix):len(r_url)]
            pth = tmp.partition("/")
            path = pth[1] + pth[2]
            words_raw, words_raw_host, words_raw_path = words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
            tld = extracted_domain.suffix
            parsed = urlparse(r_url)
            scheme = parsed.scheme

            with requests_sem:
                cert, time_cert = ef.get_cert(domain)

            extraction_data, extraction_data_time = extract_data_from_URL(hostname, content, domain, r_url)

            if type(extraction_data) == tuple:
                (Href, Link, Anchor, Media, Img, Form, CSS, Favicon, IFrame, SCRIPT, Title, Text, internals_script_doc,
                 externals_script_doc) = extraction_data
            else:
                return None

            content_di = cf.get_html_from_js(cf.remove_JScomments(internals_script_doc))
            content_de = cf.get_html_from_js(cf.remove_JScomments(externals_script_doc))

            extraction_data_di, extraction_data_di_time = extract_data_from_URL(
                hostname, content_di, domain, r_url)

            if type(extraction_data_di) == tuple:
                (Href_di, Link_di, Anchor_di, Media_di, Img_di, Form_di, CSS_di, Favicon_di, IFrame_di, SCRIPT_di,
                 Title_di,
                 Text_di, internals_script_doc_di, externals_script_doc_di) = extraction_data_di
            else:
                return None

            extraction_data_de, extraction_data_de_time = extract_data_from_URL(
                hostname, content_de, domain, r_url)

            if type(extraction_data_de) == tuple:
                (Href_de, Link_de, Anchor_de, Media_de, Img_de, Form_de, CSS_de, Favicon_de, IFrame_de, SCRIPT_de,
                 Title_de,
                 Text_de, internals_script_doc_de, externals_script_doc_de) = extraction_data_de
            else:
                return None

            lang = check_Language(content)

            with OCR_sem:
                start = time.time()
                internals_img_txt = cf.image_to_text(Img['internals'], lang)
                externals_img_txt = cf.image_to_text(Img['externals'], lang)
                extracting_ImgTxt_time = time.time() - start

                vsd = uf.count_visual_similarity_domains(second_level_domain)

            start = time.time()
            iImgTxt_words = clear_text(tokenize(internals_img_txt.lower()))
            eImgTxt_words = clear_text(tokenize(externals_img_txt.lower()))

            url_words = uf.tokenize_url(words_raw)
            sContent_words = clear_text(tokenize(Text.lower()))
            diContent_words = clear_text(tokenize(Text_di.lower()))
            deContent_words = clear_text(tokenize(Text_de.lower()))

            Text_words = iImgTxt_words + eImgTxt_words + sContent_words + diContent_words + deContent_words
            TF = compute_tf(Text_words)
            if Text_words:
                word_ratio = len(TF) / len(Text_words)
            else:
                word_ratio = 0
            preparing_words_time = time.time() - start

            iUrl_s = Href['internals'] + Link['internals'] + Media['internals'] + Form['internals']
            eUrl_s = Href['externals'] + Link['externals'] + Media['externals'] + Form['externals']
            nUrl_s = Href['null'] + Link['null'] + Media['null'] + Form['null']

            with requests_sem:
                reqs_iData_s, reqs_iTime_s = cf.get_reqs_data(iUrl_s)
                reqs_eData_s, reqs_eTime_s = cf.get_reqs_data(eUrl_s)

            iUrl_di = Href_di['internals'] + Link_di['internals'] + Media_di['internals'] + Form_di['internals']
            eUrl_di = Href_di['externals'] + Link_di['externals'] + Media_di['externals'] + Form_di['externals']
            nUrl_di = Href_di['null'] + Link_di['null'] + Media_di['null'] + Form_di['null']

            with requests_sem:
                reqs_iData_di, reqs_iTime_di = cf.get_reqs_data(iUrl_di)
                reqs_eData_di, reqs_eTime_di = cf.get_reqs_data(eUrl_di)

            iUrl_de = Href_de['internals'] + Link_de['internals'] + Media_de['internals'] + Form_de['internals']
            eUrl_de = Href_de['externals'] + Link_de['externals'] + Media_de['externals'] + Form_de['externals']
            nUrl_de = Href_de['null'] + Link_de['null'] + Media_de['null'] + Form_de['null']

            with requests_sem:
                reqs_iData_de, reqs_iTime_de = cf.get_reqs_data(iUrl_de)
                reqs_eData_de, reqs_eTime_de = cf.get_reqs_data(eUrl_de)

            with run_sem:
                record = {
                    'url': url,
                    'domain': domain,
                    'lang': lang,
                    'TF': TF,
                    'status': status,
                    'extraction-contextData-time': extraction_data_time + extraction_data_di_time + extraction_data_de_time,
                    'image-recognition-time': extracting_ImgTxt_time,
                    'stats':
                        [
                            (word_ratio, preparing_words_time),

                            #   URL FEATURES

                            uf.having_ip_address(url),
                            uf.shortening_service(url),

                            (int(cert != None), time_cert),
                            ef.good_netloc(netloc),

                            uf.url_length(r_url),

                            uf.count_at(r_url),
                            uf.count_exclamation(r_url),
                            uf.count_plust(r_url),
                            uf.count_sBrackets(r_url),
                            uf.count_rBrackets(r_url),
                            uf.count_comma(r_url),
                            uf.count_dollar(r_url),
                            uf.count_semicolumn(r_url),
                            uf.count_space(r_url),
                            uf.count_and(r_url),
                            uf.count_double_slash(r_url),
                            uf.count_slash(r_url),
                            uf.count_equal(r_url),
                            uf.count_percentage(r_url),
                            uf.count_question(r_url),
                            uf.count_underscore(r_url),
                            uf.count_hyphens(r_url),
                            uf.count_dots(r_url),
                            uf.count_colon(r_url),
                            uf.count_star(r_url),
                            uf.count_or(r_url),
                            uf.count_tilde(r_url),
                            uf.count_http_token(r_url),

                            uf.https_token(scheme),

                            uf.ratio_digits(r_url),
                            uf.count_digits(r_url),

                            cf.count_phish_hints(url_words, phish_hints, 'en'),
                            (len(url_words), 0),

                            uf.tld_in_path(tld, path),
                            uf.tld_in_subdomain(tld, subdomain),
                            uf.tld_in_bad_position(tld, subdomain, path),
                            uf.abnormal_subdomain(r_url),

                            uf.count_redirection(request),
                            uf.count_external_redirection(request, domain),

                            uf.random_domain(second_level_domain),

                            uf.random_words(words_raw),
                            uf.random_words(words_raw_host),
                            uf.random_words(words_raw_path),

                            uf.char_repeat(words_raw),
                            uf.char_repeat(words_raw_host),
                            uf.char_repeat(words_raw_path),

                            uf.punycode(r_url),
                            uf.domain_in_brand(second_level_domain),
                            uf.brand_in_path(second_level_domain, words_raw_path),
                            uf.check_www(words_raw),
                            uf.check_com(words_raw),

                            uf.port(r_url),

                            uf.length_word_raw(words_raw),
                            uf.average_word_length(words_raw),
                            uf.longest_word_length(words_raw),
                            uf.shortest_word_length(words_raw),

                            uf.prefix_suffix(r_url),

                            uf.count_subdomain(r_url),

                            vsd,

                            #   CONTENT FEATURE
                            #       (static)

                            cf.compression_ratio(request),
                            cf.count_textareas(content),
                            cf.ratio_js_on_html(Text),

                            (len(iUrl_s) + len(eUrl_s), reqs_iTime_s + reqs_eTime_s),
                            (cf.urls_ratio(iUrl_s, iUrl_s + eUrl_s + nUrl_s), reqs_iTime_s),
                            (cf.urls_ratio(eUrl_s, iUrl_s + eUrl_s + nUrl_s), reqs_eTime_s),
                            (cf.urls_ratio(nUrl_s, iUrl_s + eUrl_s + nUrl_s), 0),
                            cf.ratio_List(CSS, 'internals'),
                            cf.ratio_List(CSS, 'externals'),
                            cf.ratio_List(CSS, 'embedded'),
                            cf.ratio_List(SCRIPT, 'internals'),
                            cf.ratio_List(SCRIPT, 'externals'),
                            cf.ratio_List(SCRIPT, 'embedded'),
                            cf.ratio_List(Img, 'externals'),
                            cf.ratio_List(Img, 'internals'),
                            cf.count_reqs_redirections(reqs_iData_s),
                            cf.count_reqs_redirections(reqs_eData_s),
                            cf.count_reqs_error(reqs_iData_s),
                            cf.count_reqs_error(reqs_eData_s),
                            cf.login_form(Form),
                            cf.ratio_List(Favicon, 'externals'),
                            cf.ratio_List(Favicon, 'internals'),
                            cf.submitting_to_email(Form),
                            cf.ratio_List(Media, 'internals'),
                            cf.ratio_List(Media, 'externals'),
                            cf.empty_title(Title),
                            cf.ratio_anchor(Anchor, 'unsafe'),
                            cf.ratio_anchor(Anchor, 'safe'),
                            cf.ratio_List(Link, 'internals'),
                            cf.ratio_List(Link, 'externals'),
                            cf.iframe(IFrame),
                            cf.onmouseover(content),
                            cf.popup_window(content),
                            cf.right_clic(content),
                            cf.domain_in_text(second_level_domain, Text),
                            cf.domain_in_text(second_level_domain, Title),
                            cf.domain_with_copyright(domain, content),
                            cf.count_phish_hints(Text, phish_hints, lang),
                            (len(sContent_words), 0),
                            cf.ratio_Txt(iImgTxt_words + eImgTxt_words, sContent_words),
                            cf.ratio_Txt(iImgTxt_words, sContent_words),
                            cf.ratio_Txt(eImgTxt_words, sContent_words),
                            cf.ratio_Txt(eImgTxt_words, iImgTxt_words),

                            #       (dynamic)

                            cf.ratio_dynamic_html(Text, "".join([Text_di, Text_de])),

                            #       (dynamic internals)

                            cf.ratio_dynamic_html(Text, Text_di),
                            cf.ratio_js_on_html(Text_di),
                            cf.count_textareas(content_di),

                            (len(iUrl_di) + len(eUrl_di), reqs_iTime_de + reqs_eTime_de),
                            (cf.urls_ratio(iUrl_di, iUrl_di + eUrl_di + nUrl_di + nUrl_di), reqs_iTime_di),
                            (cf.urls_ratio(eUrl_di, iUrl_di + eUrl_di + nUrl_di), reqs_eTime_di),
                            (cf.urls_ratio(nUrl_di, iUrl_di + eUrl_di + nUrl_di), 0),
                            cf.ratio_List(CSS_di, 'internals'),
                            cf.ratio_List(CSS_di, 'externals'),
                            cf.ratio_List(CSS_di, 'embedded'),
                            cf.ratio_List(SCRIPT_di, 'internals'),
                            cf.ratio_List(SCRIPT_di, 'externals'),
                            cf.ratio_List(SCRIPT_di, 'embedded'),
                            cf.ratio_List(Img_di, 'externals'),
                            cf.ratio_List(Img_di, 'internals'),
                            cf.count_reqs_redirections(reqs_iData_di),
                            cf.count_reqs_redirections(reqs_iData_di),
                            cf.count_reqs_error(reqs_iData_di),
                            cf.count_reqs_error(reqs_iData_di),
                            cf.login_form(Form_di),
                            cf.ratio_List(Favicon_di, 'externals'),
                            cf.ratio_List(Favicon_di, 'internals'),
                            cf.submitting_to_email(Form_di),
                            cf.ratio_List(Media_di, 'internals'),
                            cf.ratio_List(Media_di, 'externals'),
                            cf.empty_title(Title_di),
                            cf.ratio_anchor(Anchor_di, 'unsafe'),
                            cf.ratio_anchor(Anchor_di, 'safe'),
                            cf.ratio_List(Link_di, 'internals'),
                            cf.ratio_List(Link_di, 'externals'),
                            cf.iframe(IFrame_di),
                            cf.onmouseover(content_di),
                            cf.popup_window(content_di),
                            cf.right_clic(content_di),
                            cf.domain_in_text(second_level_domain, Text_di),
                            cf.domain_in_text(second_level_domain, Title_di),
                            cf.domain_with_copyright(domain, content_di),

                            cf.count_io_commands(internals_script_doc),
                            cf.count_phish_hints(Text_di, phish_hints, lang),
                            (len(diContent_words), 0),

                            #       (dynamic externals)

                            cf.ratio_dynamic_html(Text, Text_de),
                            cf.ratio_js_on_html(Text_de),
                            cf.count_textareas(content_de),

                            (len(iUrl_de) + len(eUrl_de), reqs_iTime_de + reqs_eTime_de),
                            (cf.urls_ratio(iUrl_de, iUrl_de + eUrl_de + nUrl_de), reqs_iTime_de),
                            (cf.urls_ratio(eUrl_de, iUrl_de + eUrl_de + nUrl_de), reqs_eTime_de),
                            (cf.urls_ratio(nUrl_de, iUrl_de + eUrl_de + nUrl_de), 0),
                            cf.ratio_List(CSS_de, 'internals'),
                            cf.ratio_List(CSS_de, 'externals'),
                            cf.ratio_List(CSS_de, 'embedded'),
                            cf.ratio_List(SCRIPT_de, 'internals'),
                            cf.ratio_List(SCRIPT_de, 'externals'),
                            cf.ratio_List(SCRIPT_de, 'embedded'),
                            cf.ratio_List(Img_de, 'externals'),
                            cf.ratio_List(Img_de, 'internals'),
                            cf.count_reqs_redirections(reqs_iData_de),
                            cf.count_reqs_redirections(reqs_eData_de),
                            cf.count_reqs_error(reqs_iData_de),
                            cf.count_reqs_error(reqs_eData_de),
                            cf.login_form(Form_de),
                            cf.ratio_List(Favicon_de, 'externals'),
                            cf.ratio_List(Favicon_de, 'internals'),
                            cf.submitting_to_email(Form_de),
                            cf.ratio_List(Media_de, 'internals'),
                            cf.ratio_List(Media_de, 'externals'),
                            cf.empty_title(Title_de),
                            cf.ratio_anchor(Anchor_de, 'unsafe'),
                            cf.ratio_anchor(Anchor_de, 'safe'),
                            cf.ratio_List(Link_de, 'internals'),
                            cf.ratio_List(Link_de, 'externals'),
                            cf.iframe(IFrame_de),
                            cf.onmouseover(content_de),
                            cf.popup_window(content_de),
                            cf.right_clic(content_de),
                            cf.domain_in_text(second_level_domain, Text_de),
                            cf.domain_in_text(second_level_domain, Title_de),
                            cf.domain_with_copyright(domain, content_de),

                            cf.count_io_commands(externals_script_doc),
                            cf.count_phish_hints(Text_de, phish_hints, lang),
                            (len(deContent_words), 0),

                            #   EXTERNAL FEATURES

                            ef.domain_registration_length(domain),
                            # do not use VPN: Error trying to connect to socket: closing socket
                            ef.whois_registered_domain(domain),
                            ef.web_traffic(r_url),
                            ef.page_rank(domain),
                            ef.remainder_valid_cert(cert),
                            ef.valid_cert_period(cert),
                            ef.count_alt_names(cert)
                        ]
                }

            return record
        return None

    log_event = threading.Event()
    work_event = threading.Event()
    save_obj = threading.Lock()


    def generate_dataset(url_list):
        def init():
            h1 = headers['metadata'] + headers['substats'] + headers['stats']
            h2 = headers['metadata'] + headers['stats'] + ['TF']

            pandas.DataFrame(np.array(h1).reshape(1, len(h1))).to_csv(dir_path + 'feature_times.csv', index=False,
                                                                      header=False)
            pandas.DataFrame(np.array(h2).reshape(1, len(h2))).to_csv(dir_path + 'dataset.csv', index=False,
                                                                      header=False)

        init()

        def extraction_data(obj):
            try:
                res = extract_features(obj[0], obj[1])

                if type(res) is dict:
                    with save_obj:
                        tmp = res['stats']
                        metadata = [res[key] for key in headers['metadata']]
                        substats = [res[key] for key in headers['substats']]
                        TF = ';'.join(['{}={}'.format(k, res['TF'][k]) for k in res['TF']])
                        data = metadata + [data[0] for data in tmp] + [TF]
                        counter = metadata + substats + [data[1] for data in tmp]

                        pandas.DataFrame(data).T.to_csv(dir_path + 'dataset.csv', mode='a',
                                                        index=False, header=False)
                        pandas.DataFrame(counter).T.to_csv(dir_path + 'feature_times.csv',
                                                           mode='a',
                                                           index=False, header=False)
            except Exception:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=extractor_thread_count) as executor:
            # executor.map(extraction_data, url_list)
            fut = [executor.submit(extraction_data, url) for url in url_list]
            for _ in tqdm(concurrent.futures.as_completed(fut), total=len(url_list)):
                pass


if state == 1:
    from random import randint
    from datetime import date

    filter1 = re.compile('"|javascript:|void(0)|((\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$)').search
    filter2 = re.compile('(\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$').search

    def extract_URLs_from_page(hostname, content, domain):
        Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                       "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

        Href = []
        soup = BeautifulSoup(content, 'html.parser')

        # collect all external and internal hrefs from url
        for href in soup.find_all('a', href=True):
            dots = [x.start(0) for x in re.finditer('\.', href['href'])]
            if hostname in href['href'] or domain in href['href'] or len(dots) == 1 or not href['href'].startswith(
                    'http'):
                if not href['href'].startswith('http'):
                    if not href['href'].startswith('/'):
                        Href.append('http://' + hostname + '/' + href['href'])
                    elif href['href'] not in Null_format:
                        Href.append('http://' + hostname + href['href'])
            else:
                Href.append(href['href'])

        Href = [url for url in Href if url if url and not filter1(url) and not filter2(urlparse(url).path)]

        return Href

    def find_duplicates(list):
        class Dictlist(dict):
            def __setitem__(self, key, value):
                try:
                    self[key]
                except KeyError:
                    super(Dictlist, self).__setitem__(key, [])
                self[key].append(value)

        dom_urls = Dictlist()

        for url in tqdm(list, total=len(list)):
            dom_urls[urlparse(url).netloc] = url

        list = []

        for item in tqdm(dom_urls.values(), total=len(dom_urls.values())):
            list.append(item[randint(0, len(item)-1)])

        return list

    def filter_url_list(url_list):
        return [url for url in url_list if
                not re.search('"|javascript:|void(0)|((\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$)|{|}', url) and
                not re.search('(\.js|\.png|\.css|\.ico|\.jpg|\.json|\.csv|\.xml|/#)$', urlparse(url).path)]

    def generate_legitimate_urls(N):
        domain_list = pandas.read_csv("data/ranked_domains/14-1-2021.csv",
                                      header=None)[1].tolist()

        url_list = []

        def url_search(domain):
            if len(url_list) >= N:
                return

            url = search_for_vulnerable_URLs(domain)

            if url:
                url_list.append(url)

                if len(url_list) % 10 == 0:
                    pandas.DataFrame(url_list).to_csv(
                        "data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")), index=False,
                        header=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            fut = [executor.submit(url_search, domain) for domain in domain_list]
            for _ in tqdm(concurrent.futures.as_completed(fut), total=len(domain_list)):
                pass

        url_list = find_duplicates(filter_url_list(url_list))
        pandas.DataFrame(url_list).to_csv("data/urls/legitimate/{0}.csv".format(date.today().strftime("%d-%m-%Y")),
                                          index=False, header=False)

    def search_for_vulnerable_URLs(domain):
        url = 'http://' + domain

        state, request = is_URL_accessible(url)[0]

        if state:
            url = request.url
            content = request.content
            hostname, domain, path, netloc = get_domain(url)
            extracted_domain = tldextract.extract(url)
            domain = extracted_domain.domain + '.' + extracted_domain.suffix

            Href = extract_URLs_from_page(hostname, content, domain)

            if Href:
                url = Href[randint(0, len(Href))-1]
                state, request = is_URL_accessible(url)[0]
                if state:
                    return request.url

        return None


if state == 7:
    def combine_datasets():
        lst_files = list(map(int, os.listdir(os.getcwd() + '/data/datasets/RAW')))
        lst_files.sort()


        df = [pandas.read_csv("data/datasets/RAW/{}/dataset.csv".format(i), index_col='id') for i in lst_files]
        frame = pandas.concat(df, axis=0)
        frame.drop_duplicates(subset=['url'], keep='last')


        frame = frame.drop(['TF','кол-во поддоменов'], axis=1)

        frame = frame.loc[frame['срок регистрации домена'] <= 10000]

        frame.loc[frame['степень сжатия страницы'] > 1, 'степень сжатия страницы'] = 1

        frame['рейтинг по Alexa'] = frame['рейтинг по Alexa']/10000000
        frame.loc[frame['рейтинг по Alexa'] > 1, 'рейтинг по Alexa'] = 1

        frame = frame.replace(-5, 0)

        frame = frame.reset_index(drop=True)

        frame.to_csv("data/datasets/PROCESS/dataset.csv", index_label='id')

        frame = frame.drop(headers['metadata'], axis=1)

        head = list(frame)
        max = frame.max().to_list()
        min = frame.min().to_list()
        mean = frame.mean().to_list()
        pandas.DataFrame([head, max, min, mean]).T.to_csv("data/datasets/PROCESS/dataset_stats.csv", index=False,
                                                          header=['feature', 'max', 'min', 'mean'])

        df = [pandas.read_csv("data/datasets/RAW/{}/feature_times.csv".format(i)) for i in lst_files]
        frame = pandas.concat(df, axis=0)

        head = list(frame)
        max = frame.max().to_list()
        min = frame.min().to_list()
        mean = frame.mean().to_list()
        pandas.DataFrame([head, max, min, mean]).T.to_csv("data/datasets/PROCESS/feature_times.csv", index=False,
                                                        header=['feature', 'max', 'min', 'mean'])


if state == 8 or state == 48:
    import numpy as np
    from sklearn import svm

    from feature_selector import FeatureSelector


    def RFE(X, Y, N, step=10):
        clfRFE = svm.SVC(kernel='linear', cache_size=4096)
        featureCount = X.shape[1]
        featureList = np.arange(0, featureCount)
        included = np.full(featureCount, True)
        curCount = featureCount
        while curCount > N:
            actualFeatures = featureList[included]
            Xnew = X.iloc[:, actualFeatures]

            clfRFE.fit(Xnew, Y)
            curStep = min(step, curCount - N)
            elim = np.argsort(np.abs(clfRFE.coef_[0]))[:curStep]
            included[actualFeatures[elim]] = False
            curCount -= curStep

        included = [idx for idx in featureList if included[idx]]
        return X.iloc[:, included]


    def select_features(N):
        frame = pandas.read_csv('data/datasets/OUTPUT/dataset.csv')

        # нормализация

        cols = [col for col in headers['stats'] if col in list(frame)]

        X = frame[cols]
        Y = frame['status']

        X = (X - X.min()) / (X.max() - X.min())

        fs = FeatureSelector(data=X, labels=Y)

        fs.identify_all({
            'missing_threshold': 0.6,
            'correlation_threshold': 0.98,
            'eval_metric': 'auc',
            'task': 'classification',
            'cumulative_importance': 0.99
        })

        fs.plot_feature_importances(threshold=0.99, plot_n=10)
        selected = fs.remove(methods='all')

        X = selected

        # сокращение числа параметров


        X = RFE(X, Y, N)

        max = X.max().to_list()
        min = X.min().to_list()
        mean = X.mean().to_list()
        pandas.DataFrame([list(X), max, min, mean]).T.to_csv("data/datasets/OUTPUT2/dataset_stats.csv", index=False,
                                                             header=['feature', 'max', 'min', 'mean'])

        X['status'] = Y
        X.to_csv("data/datasets/OUTPUT2/dataset.csv", index=False)
