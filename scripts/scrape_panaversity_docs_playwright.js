#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const START_URL = 'https://agentfactory.panaversity.org/docs/about';
const ORIGIN = 'https://agentfactory.panaversity.org';
const DOCS_PREFIX = '/docs/';

function parseArgs(argv) {
  const out = {
    outDir: 'data/panaversity_scrape_playwright',
    delayMs: 80,
    maxPages: 0,
    startSlug: '',
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--out-dir' && argv[i + 1]) {
      out.outDir = argv[i + 1];
      i += 1;
    } else if (arg === '--delay-ms' && argv[i + 1]) {
      out.delayMs = Math.max(0, Number(argv[i + 1]) || 0);
      i += 1;
    } else if (arg === '--max-pages' && argv[i + 1]) {
      out.maxPages = Math.max(0, Number(argv[i + 1]) || 0);
      i += 1;
    } else if (arg === '--start-slug' && argv[i + 1]) {
      out.startSlug = String(argv[i + 1] || '').trim();
      i += 1;
    } else if (arg === '--help' || arg === '-h') {
      console.log('Usage: node scripts/scrape_panaversity_docs_playwright.js [--out-dir DIR] [--delay-ms N] [--max-pages N] [--start-slug SLUG]');
      process.exit(0);
    }
  }

  return out;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function slugify(value) {
  const out = value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return out || 'doc';
}

function normalizeDocUrl(rawHref) {
  const abs = new URL(rawHref, ORIGIN);
  if (!abs.pathname.startsWith(DOCS_PREFIX)) {
    return null;
  }
  abs.hash = '';
  abs.search = '';
  return abs.toString();
}

function pathParts(urlStr) {
  const u = new URL(urlStr);
  const trimmed = u.pathname.slice(DOCS_PREFIX.length).replace(/^\/+|\/+$/g, '');
  const segments = trimmed ? trimmed.split('/').map((s) => decodeURIComponent(s)) : [];
  return { pathname: u.pathname, segments };
}

async function collectSidebarOrder(page) {
  await page.goto(START_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('aside nav[aria-label="Docs sidebar"]', { timeout: 30000 });

  for (let i = 0; i < 20; i += 1) {
    const remaining = await page.locator('aside .menu__caret[aria-expanded="false"]').count();
    if (remaining === 0) {
      break;
    }
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('aside .menu__caret[aria-expanded="false"]'));
      for (const btn of buttons) {
        btn.click();
      }
    });
    await page.waitForTimeout(200);
  }

  const orderedLinks = await page.evaluate(() => {
    const nav = document.querySelector('aside nav[aria-label="Docs sidebar"]');
    if (!nav) {
      return [];
    }

    const seen = new Set();
    const rows = [];
    const anchors = Array.from(nav.querySelectorAll('a[href^="/docs/"]'));

    for (const a of anchors) {
      const href = a.getAttribute('href') || '';
      const abs = new URL(href, location.origin).toString();
      if (seen.has(abs)) {
        continue;
      }
      seen.add(abs);

      const text = (a.textContent || '').replace(/\s+/g, ' ').trim();
      const li = a.closest('li');
      const classes = li ? Array.from(li.classList) : [];
      const levelClass = classes.find((c) => c.includes('level-')) || '';
      const levelMatch = levelClass.match(/level-(\d+)/);
      const level = levelMatch ? Number(levelMatch[1]) : null;
      const type = classes.some((c) => c.includes('item-category')) ? 'category' : 'link';

      rows.push({ href: abs, title: text, level, type });
    }

    return rows;
  });

  return orderedLinks
    .map((row) => {
      const normalized = normalizeDocUrl(row.href);
      if (!normalized) {
        return null;
      }
      return {
        ...row,
        href: normalized,
      };
    })
    .filter(Boolean);
}

async function scrapeDocs(opts) {
  const outDir = path.resolve(opts.outDir);
  const rawDir = path.join(outDir, 'raw_html');
  const txtDir = path.join(outDir, 'content');
  ensureDir(outDir);
  ensureDir(rawDir);
  ensureDir(txtDir);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (compatible; PanaversityDocsScraperPlaywright/1.0)',
  });
  const page = await context.newPage();

  const sidebarRows = await collectSidebarOrder(page);
  let orderedRows = sidebarRows;
  if (opts.startSlug) {
    const wantedPath = `${DOCS_PREFIX}${opts.startSlug.replace(/^\/+|\/+$/g, '')}`;
    const startIndex = orderedRows.findIndex((row) => new URL(row.href).pathname === wantedPath);
    if (startIndex >= 0) {
      orderedRows = orderedRows.slice(startIndex);
    }
  }
  const limitedRows = opts.maxPages > 0 ? orderedRows.slice(0, opts.maxPages) : orderedRows;

  fs.writeFileSync(path.join(outDir, 'sidebar_links.json'), JSON.stringify(sidebarRows, null, 2));
  fs.writeFileSync(path.join(outDir, 'docs_urls.txt'), `${limitedRows.map((r) => r.href).join('\n')}\n`);

  const records = [];
  const failures = [];

  for (let i = 0; i < limitedRows.length; i += 1) {
    const row = limitedRows[i];
    const seq = i + 1;
    try {
      await page.goto(row.href, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.waitForSelector('article', { timeout: 30000 });

      const title = await page.locator('article h1').first().textContent().catch(() => null);
      const docTitle = (title || '').trim() || (await page.title()).replace(/\s*\|\s*Agent Factory\s*$/i, '').trim();

      const contentText = await page
        .locator('article .theme-doc-markdown.markdown')
        .first()
        .innerText()
        .catch(() => '');

      const html = await page.content();
      const { pathname, segments } = pathParts(row.href);
      const part = segments.length >= 1 ? segments[0] : null;
      const chapter = segments.length >= 2 ? segments[1] : null;
      const lesson = segments.length >= 3 ? segments.slice(2).join('/') : null;
      const depth = segments.length;

      const safeName = slugify(segments.join('-') || 'about');
      const rawFile = `${String(seq).padStart(4, '0')}-${safeName}.html`;
      const txtFile = `${String(seq).padStart(4, '0')}-${safeName}.txt`;

      fs.writeFileSync(path.join(rawDir, rawFile), html);
      fs.writeFileSync(path.join(txtDir, txtFile), `${contentText.trim()}\n`);

      records.push({
        sequence: seq,
        url: row.href,
        path: pathname,
        title: docTitle,
        sidebar_title: row.title,
        sidebar_level: row.level,
        sidebar_type: row.type,
        part_slug: part,
        chapter_slug: chapter,
        lesson_slug: lesson,
        depth,
        text_file: path.join('content', txtFile),
        source_file: path.join('raw_html', rawFile),
      });
    } catch (err) {
      failures.push({
        sequence: String(seq),
        url: row.href,
        error: String(err && err.message ? err.message : err),
      });
    }

    if (opts.delayMs > 0) {
      await sleep(opts.delayMs);
    }
  }

  await context.close();
  await browser.close();

  const hierarchy = {};
  for (const r of records) {
    const part = r.part_slug || '_root';
    const chapter = r.chapter_slug || '_intro';
    if (!hierarchy[part]) {
      hierarchy[part] = {};
    }
    if (!hierarchy[part][chapter]) {
      hierarchy[part][chapter] = [];
    }
    if (r.lesson_slug) {
      hierarchy[part][chapter].push(r.lesson_slug);
    }
  }

  const index = {
    source: START_URL,
    scraped_at_unix: Math.floor(Date.now() / 1000),
    total_sidebar_urls: sidebarRows.length,
    total_docs_urls: limitedRows.length,
    success_count: records.length,
    failure_count: failures.length,
    records,
  };

  fs.writeFileSync(path.join(outDir, 'docs_index.json'), JSON.stringify(index, null, 2));
  fs.writeFileSync(path.join(outDir, 'failures.json'), JSON.stringify(failures, null, 2));
  fs.writeFileSync(path.join(outDir, 'hierarchy.json'), JSON.stringify(hierarchy, null, 2));

  console.log(`Sidebar links found: ${sidebarRows.length}`);
  console.log(`Scraped pages: ${records.length}`);
  console.log(`Failures: ${failures.length}`);
  console.log(`Output: ${outDir}`);
}

(async () => {
  const opts = parseArgs(process.argv);
  await scrapeDocs(opts);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
