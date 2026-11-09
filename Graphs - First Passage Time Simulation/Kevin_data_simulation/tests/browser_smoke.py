"""Optional real-browser workflow check. Requires Playwright + Chromium.

Run from the project root: python tests/browser_smoke.py
Uses a temporary run store, so it does not add test runs to the user's notebook.
"""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import RunStore, make_handler
from playwright.sync_api import expect, sync_playwright


def main():
    with tempfile.TemporaryDirectory() as folder:
        store=RunStore(Path(folder)/'runs')
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(store))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        errors=[]
        try:
            with sync_playwright() as playwright:
                browser=playwright.chromium.launch(headless=True)
                context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
                page=context.new_page()
                page.on('pageerror',lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
                expect(page.locator('#history-body')).to_contain_text('No experiments yet')
                expect(page.locator('#graph-description')).to_contain_text('2 × 4')
                page.locator('#name').fill('Browser check · ladder')
                page.locator('#N').fill('2000')
                page.locator('#run-button').click()
                expect(page.locator('#result-name')).to_have_text('Browser check · ladder',timeout=15000)
                expect(page.locator('#metric-frne')).to_contain_text('%')
                page.locator('#source').select_option('3');page.locator('#target').select_option('3')
                expect(page.locator('#pair-note')).to_contain_text('First return')
                page.locator('#view-cdf').click()
                expect(page.locator('#view-cdf')).to_have_attribute('aria-pressed','true')
                expect(page.locator('#ci-legend')).to_be_hidden()
                page.locator('#view-pmf').click()
                with page.expect_download() as download:
                    page.locator('#export-json').click()
                result=json.loads(Path(download.value.path()).read_text())
                assert result['config']['N']==2000 and len(result['r_exact'])==17
                with page.expect_download() as download:
                    page.locator('#export-csv').click()
                assert 'r_exact,r_empirical' in Path(download.value.path()).read_text()
                page.locator('#graph').select_option('paper9')
                page.locator('#q-mode').select_option('uniform');page.locator('#uniform-q').fill('.2')
                page.locator('.advanced summary').click();page.locator('#dt').fill('1')
                page.locator('#run-button').click()
                expect(page.locator('#error')).to_be_visible()
                expect(page.locator('#error')).to_contain_text('dt must be')
                page.locator('#dt').fill('auto');page.locator('#q-mode').select_option('vertex')
                page.locator('#q-4').fill('.4');page.locator('#name').fill('Browser check · potential')
                page.locator('#T').fill('12');page.locator('#N').fill('3000')
                with page.expect_download() as download:
                    page.locator('#save-config').click()
                saved=json.loads(Path(download.value.path()).read_text())
                assert saved['q']==[0,0,0,0,.4,0,0,0,0]
                page.locator('#run-button').click()
                expect(page.locator('#result-name')).to_have_text('Browser check · potential',timeout=15000)
                expect(page.locator('#result-config')).to_contain_text('h=0.714286')
                expect(page.locator('#error')).to_be_hidden()
                page.locator('#data-details summary').click()
                expect(page.locator('#data-table tbody tr')).to_have_count(12)
                imported={'name':'Single-step check','graph':'paper9','T':1,'N':800,'q':0,'seed':7,'dt':'auto'}
                page.locator('#import-config').set_input_files({'name':'config.json','mimeType':'application/json','buffer':json.dumps(imported).encode()})
                expect(page.locator('#name')).to_have_value('Single-step check')
                page.locator('#run-button').click()
                expect(page.locator('#result-name')).to_have_text('Single-step check',timeout=15000)
                page.locator('#source').select_option('0');page.locator('#target').select_option('0')
                expect(page.locator('#pair-note')).to_contain_text('FRNE: undefined')
                page.set_viewport_size({'width':390,'height':844})
                assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), 'Mobile horizontal overflow'
                page.set_viewport_size({'width':1440,'height':1100})
                page.locator('#paper-preset').click()
                expect(page.locator('#T')).to_have_value('17')
                expect(page.locator('#N')).to_have_value('1000000')
                expect(page.locator('#q-mode')).to_have_value('zero')
                page.locator('#name').fill('Cancellation check');page.locator('#T').fill('40')
                page.locator('#run-button').click()
                expect(page.locator('#progress-panel')).to_be_visible()
                page.locator('#cancel-run').click()
                expect(page.locator('#history-body')).to_contain_text('cancelled',timeout=15000)
                expect(page.locator('#progress-panel')).to_be_hidden()
                page.locator('#history-body tr').filter(has_text='Browser check · potential').get_by_role('button',name='Reuse').click()
                expect(page.locator('#name')).to_have_value('Browser check · potential')
                expect(page.locator('#q-4')).to_have_value('0.4')
                page.reload(wait_until='networkidle')
                expect(page.locator('#history-body')).to_contain_text('Browser check · potential')
                page.locator('#history-body tr').filter(has_text='Browser check · potential').get_by_role('button',name='View',exact=True).click()
                expect(page.locator('#result-name')).to_have_text('Browser check · potential')
                assert not errors, errors
                browser.close()
            print('PASS: run, returns, PMF/CDF, JSON/CSV, invalid dt, vertex q, config export/import, T=1, mobile layout, paper preset, cancellation, reuse, history reload; no browser exceptions.')
        finally:
            server.shutdown();server.server_close();thread.join();store.close()


if __name__=='__main__':
    main()
