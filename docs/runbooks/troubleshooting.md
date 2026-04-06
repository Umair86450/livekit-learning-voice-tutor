# Troubleshooting

## Common Issues
- Missing `GROQ_API_KEY` -> runtime/config failure.
- STT local mode slow or failing -> verify whisper model exists.
- RAG empty results -> verify Qdrant reachable and data prepared.
- Test failures after prompt/config edits -> run focused test files first.

## Fast Debug Path
1. Re-run with focused command.
2. Check env values.
3. Verify required files exist (`models/`, prepared data).
4. Run targeted pytest for changed area.
