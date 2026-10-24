# Fonts

The UI loads exactly two local fonts, both through `next/font/local` in `src/app/layout.tsx`:

| File | Used for | Source and licence |
|---|---|---|
| `Satoshi-Variable.woff2` | all text (`--font-satoshi`; Tailwind `sans`, `display`, `mono`) | Satoshi by Indian Type Foundry, from Fontshare (fontshare.com), ITF Free Font License |
| `Doto-Variable.woff2` | digits only (`--font-doto`, `unicode-range` U+0030-0039; Tailwind `dotted`, `numbers`) | Doto, from Google Fonts, SIL Open Font License 1.1 |

Other families that once sat in this folder were never referenced and were removed. Add a font
here only together with the `localFont` call that loads it.
