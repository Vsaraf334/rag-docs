# Experiment: hybrid search (vector + keyword)

**Result: not adopted.** Hybrid search scored the same retrieval hit rate as vector search (19/20) but made two answers worse. The code is kept on the `experiment/hybrid-search` branch.

```bash
git switch experiment/hybrid-search                                     # get the code back
uv run --env-file .env python src/pg_store.py                           # adds the keyword column
uv run --env-file .env python evals/run_eval.py --backend hybrid
git switch main
```

## 1. The idea

There are two ways to search text, and they fail in different places:

| | Vector search (embeddings) | Keyword search (full-text / BM25) |
| --- | --- | --- |
| Matches | Similar **meaning** | The same **words** |
| Strong at | Paraphrases: "item not found" ↔ "404" | Exact identifiers: `QX42`, `add_task`, `RequestValidationError` |
| Weak at | Rare exact codes and names | Synonyms and rephrasing |

**Hybrid search** runs both and merges the two ranked lists. A chunk ranked well by *both* methods is very likely relevant.

## 2. Keyword search in Postgres

Postgres has full-text search built in, so no extra library is needed. Four building blocks:

**`to_tsvector('english', text)`** turns a text into a list of searchable words. It drops filler words (stop words) and reduces each word to its stem:

```
'How do I restrict a URL parameter to a fixed set of allowed values?'
  → 'restrict' 'url' 'paramet' 'fix' 'set' 'allow' 'valu'
```

"how", "do", "I", "a", "to" and "of" are dropped. "values" becomes `valu`, so "value" and "values" match each other.

**`plainto_tsquery('english', question)`** turns the question into a search query in the same way. By default it joins the words with AND (`&`).

**`words @@ query`** means "does this chunk match this query?", the full-text version of `LIKE`.

**`ts_rank_cd(words, query)`** gives a relevance score: higher when more query words appear, and when they appear close together ("cd" stands for cover density).

### The column and the index

```sql
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS words tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_words_idx ON chunks USING gin (words);
```

- **`GENERATED ALWAYS AS (...) STORED`**: Postgres computes `words` from `text` itself, on every insert and update. The Python code never touches this column and can't forget to update it.
- **`USING gin`**: a GIN (generalized inverted) index maps each word to the rows that contain it, like the index at the back of a book. It's the same idea as a Lucene/Elasticsearch index, and it's what keeps `@@` fast on millions of rows.

### Gotcha: AND finds nothing for natural-language questions

```
plainto_tsquery(...)       → 'restrict' & 'url' & 'paramet' & 'fix' & 'set' & 'allow' & 'valu'
chunks matching with AND   → 0
chunks matching with OR    → 77
```

No chunk contains all seven words. The fix is to replace every `&` with `|` (OR) and let `ts_rank_cd` rank the 77 candidates:

```sql
replace(plainto_tsquery('english', question)::text, '&', '|')::tsquery
```

`::text` turns the query into a string, `replace` swaps the operators, and `::tsquery` turns it back into a query.

## 3. Merging the two lists: Reciprocal Rank Fusion (RRF)

The two searches produce scores in different units. A cosine distance of 0.43 and a `ts_rank_cd` of 0.1 can't be added. RRF ignores the scores and uses only **ranks**:

```
score(chunk) = 1 / (60 + rank in vector list) + 1 / (60 + rank in keyword list)
```

A list the chunk doesn't appear in contributes 0.

Worked example:

| Chunk | Vector rank | Keyword rank | RRF score | Final position |
| --- | --- | --- | --- | --- |
| A | 1 | 2 | 1/61 + 1/62 = 0.0325 | **1st** |
| C | 3 | 1 | 1/63 + 1/61 = 0.0323 | **2nd** |
| B | 2 | — | 1/62 + 0 = 0.0161 | 3rd |
| D | — | 3 | 0 + 1/63 = 0.0159 | 4th |

B was 2nd in the vector list, but C overtakes it because C appears in **both** lists. That's the point of RRF.

**Why 60?** It's the constant from the original RRF paper. It flattens the curve so that being 1st instead of 2nd matters only a little. Without it, 1/1 versus 1/2 would let a single list dominate.

## 4. The SQL, piece by piece

```sql
WITH vector AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s) AS rank
    FROM chunks
    ORDER BY embedding <=> %(vector)s
    LIMIT %(candidates)s
),
```

- **`WITH name AS (...)`** is a CTE (common table expression): a named temporary result you can use later in the query, like a local variable for SQL.
- **`row_number() OVER (ORDER BY ...)`** is a window function. It numbers the rows 1, 2, 3… in distance order, which turns distances into **ranks**.
- **`LIMIT %(candidates)s`** keeps the best 50. Each list only needs enough candidates to fuse.

```sql
keyword AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(words, query) DESC) AS rank
    FROM chunks,
         (SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')::tsquery
          AS query) AS q
    WHERE words @@ query
    ORDER BY ts_rank_cd(words, query) DESC
    LIMIT %(candidates)s
)
```

- The subquery `(SELECT ... AS query) AS q` builds the OR query **once**, and `FROM chunks, q` makes it available on every row.
  - Postgres doesn't allow a `::` cast directly in the `FROM` list, hence the subquery wrapper. That's the syntax error hit during the experiment.
- `WHERE words @@ query` keeps only the chunks containing at least one of the words.
- Ranking runs by `ts_rank_cd ... DESC` (higher is better), the opposite direction from distances.

```sql
SELECT c.text, c.source,
       coalesce(1.0 / (%(rrf_k)s + v.rank), 0)
     + coalesce(1.0 / (%(rrf_k)s + k.rank), 0) AS score
FROM vector v
FULL OUTER JOIN keyword k ON v.id = k.id
JOIN chunks c ON c.id = coalesce(v.id, k.id)
ORDER BY score DESC
LIMIT %(top_k)s
```

- **`FULL OUTER JOIN`** keeps chunks that appear in **either** list. A chunk found only by keyword search has `v.rank = NULL`, and vice versa. With the worked example above:

  | v.id | v.rank | k.id | k.rank |
  | --- | --- | --- | --- |
  | A | 1 | A | 2 |
  | B | 2 | NULL | NULL |
  | C | 3 | C | 1 |
  | NULL | NULL | D | 3 |

  An `INNER JOIN` would keep only A and C and lose the chunks that just one method found.
- **`1.0 / (60 + NULL)`** is `NULL`, and **`coalesce(x, 0)`** turns `NULL` into 0: "not in this list contributes nothing". `1.0` rather than `1` forces decimal division; `1 / 61` in integer maths is 0.
- **`coalesce(v.id, k.id)`** picks whichever id isn't `NULL`, and the join to `chunks c` fetches that chunk's text and source.
- **`ORDER BY score DESC LIMIT top_k`** returns the best fused results.

### The Python side

```python
params = {"vector": embed([question])[0], "question": question,
          "candidates": 50, "rrf_k": 60, "top_k": top_k}
rows = conn.execute(HYBRID_SEARCH, params).fetchall()
```

**Named placeholders** like `%(vector)s` take their values from a dict. They're used here because `%(vector)s` appears twice and `%(rrf_k)s` twice. With plain `%s`, every repeat would need its own value, in exact order. It's the same as `:name` parameters in Spring's `NamedParameterJdbcTemplate`.

## 5. Results

Same data, same model, `top_k = 8`:

| Backend | Retrieval | Answers | Refusals |
| --- | --- | --- | --- |
| Vector (pgvector) | 19/20 | 18–19/20 | 4/4 |
| Hybrid (vector + keyword, RRF) | 19/20 | 18/20 | 4/4 |

The totals look similar, but the details show hybrid is worse:

| Question | What hybrid did |
| --- | --- |
| "How do I restrict a URL parameter to a fixed set of allowed values?" | Counted as a hit, but the `path-params.md` chunk was the page **introduction**, not the `Enum` section. A hollow fix. |
| "What happens if a client sends text where the URL expects a number?" | **New miss.** "client", "text" and "number" pulled in `handling-errors.md` chunks. |
| "How do I run code after sending the response?" | Still a hit, but the 4 chunks explaining `add_task` were replaced by `response-model.md` chunks matching the common word **"response"**. The answer dropped to `partial`. |

## 6. Why it didn't help here

- **The questions are natural language**, not identifiers. Keyword search mostly matched common words ("response", "parameter", "client"), which adds noise rather than signal.
- **The remaining miss is a vocabulary mismatch.** The question says "fixed set of allowed values" and the docs say "predefined values" and "Enum". Keyword search can only match words that are in the question, so it can't bridge that gap.
- **Vector search already handled identifiers well** on this data: 9/10 on an identifier-only check run before the experiment.

Hybrid search is worth it when users search with **exact codes and names**: error codes, SKUs, function names, ticket IDs. The branch's test `test_hybrid_finds_exact_keyword_and_sorts_by_score` shows that case: a search for `QX42` returns the one chunk containing it first.

## 7. What the experiment taught

1. **Measure before adopting a "best practice".** Hybrid search is standard advice, and on this data it was a regression.
2. **Look beyond the totals.** 19/20 → 19/20 hid one fix that was hollow and one that broke something else.
3. **The file-level retrieval metric has a blind spot.** It counted the hollow Enum hit as a success and missed the background-tasks chunks being displaced. A chunk-level relevance check would catch both.
4. **Keep experiments on a branch with a written result.** The code stays runnable, and the reason it was dropped isn't forgotten.

## 8. Revision questions

1. Why can't you add a cosine distance and a `ts_rank_cd` score together, and how does RRF avoid the problem?
2. Chunk X is 1st in the vector list only; chunk Y is 5th in both lists. Which ranks higher after RRF? (X: 1/61 = 0.0164. Y: 2/65 = 0.0308. **Y.**)
3. Why does `plainto_tsquery` return 0 matches for most questions, and what's the fix?
4. What would an `INNER JOIN` instead of `FULL OUTER JOIN` lose?
5. Why `coalesce(..., 0)`, and why `1.0` instead of `1`?
6. What does `GENERATED ALWAYS AS (...) STORED` save you from?
7. What is a GIN index, and what is its Lucene equivalent?
8. Name a dataset where hybrid search would beat vector search, and explain why.
