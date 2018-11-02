from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
import click
from fuzzyfinder import fuzzyfinder
from pygments.lexers.sql import SqlLexer

SQLKeywords = ['select', 'from', 'insert', 'update', 'delete', 'drop']

class SQLCompleter(Completer):
    def get_completions(self, document, complete_event):
        word_before_cursor = document.get_word_before_cursor(WORD=True)
        matches = fuzzyfinder(word_before_cursor, SQLKeywords)
        for m in matches:
            yield Completion(m, start_position=-len(word_before_cursor))

while 1:
    user_input = prompt(u'SQL>',
                        history=FileHistory('history.txt'),
                        auto_suggest=AutoSuggestFromHistory(),
                        completer=SQLCompleter(),
                        lexer=SqlLexer,
                        )
    click.echo_via_pager(user_input)

