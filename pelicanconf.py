#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'João Oda'
SITENAME = 'Random Reasons & Reflections'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/Sao_Paulo'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
#LINKS = (('Pelican', 'http://getpelican.com/'),
#         ('Python.org', 'http://python.org/'),
#         ('Jinja2', 'http://jinja.pocoo.org/'),
#         ('You can modify those links in your config file', '#'),)

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)
SOCIAL = (('LinkedIn', 'https://www.linkedin.com/in/joão-oda-5b37549b'),
          ('Facebook', 'https://www.facebook.com/joao.oda.1'),
          ('GitHub', 'https://github.com/ojon'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

#Theme and Plug-ins
MARKUP = ('md', 'ipynb')

PLUGIN_PATHS = ['./plugins', './pelican-plugins']
PLUGINS = [
  'ipynb.markup',
  'i18n_subsites',
  'tag_cloud'
]
THEME = './pelican-themes/pelican-bootstrap3'

JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
I18N_GETTEXT_NEWSTYLE = True

I18N_SUBSITES = {
        'pt': {
            'SITENAME': 'Razões & Refleções Randômicas',
            'LOCALE': 'pt_BR.utf8',            #This is somewhat redundant with DATE_FORMATS, but IMHO more convenient
            #'STATIC_PATHS':['pt/images', 'pt/files'],
            'ABOUT_ME': '''Eu sou uma pessoa curiosa com formação multidisciplinar, interessado em resolver problemas e melhorar a vida. Utilizado de técnicas analíticas, computacionais, inteligência artificial (principalmente com o uso de dados) para encontrar uma solução. Eu gosto de ciência e atividades estimulantes (tanto do ponto de vista físico como intelectual). Procuro desenvolvimento pessoal, valorizo saúde e tento seguir um estilo de vida saudável.'''
            },
        }

#Bootstrap Customization
BOOTSTRAP_THEME = 'readable'
PYGMENTS_STYLE = 'default'
ABOUT_ME = '''I am a curious person with multidisciplinary background,
 who is interested in solve problems and enhance life.
 I employ analytical, computational, AI (mainly data driven)
 techniques to find a solution.
 I like science and stimulating activities (physical and intellectual). I search for personal development , value health and attempt to live a healthy lifestyle.  '''

AVATAR = 'images/profile.jpg'

#HIDE_SIDEBAR = True

DISPLAY_ARTICLE_INFO_ON_INDEX = True

DISPLAY_TAGS_ON_SIDEBAR = True
#DISPLAY_TAGS_INLINE = True

OUTPUT_SOURCES = True

STATIC_PATHS = ['images', 'files']

EXTRA_PATH_METADATA = {
    'files/.nojekyll': {
        'path': '.nojekyll',
        },
    }
