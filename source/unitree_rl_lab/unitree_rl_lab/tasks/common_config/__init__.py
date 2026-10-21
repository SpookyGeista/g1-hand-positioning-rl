"""Gemeinsame Konfigurations-Utilities.

Hinweis:
- Die alte Version dieses Moduls hat weitere Sub-Module (z.B. ``camera_configs``,
  ``robots.unitree``) vorausgesetzt, die im aktuellen Repository nicht vorhanden
  sind.
- Um Importfehler beim automatischen Laden aller Tasks zu vermeiden, halten wir
  dieses ``__init__`` bewusst minimal und importieren keine weiteren Module.

Falls du später wieder gemeinsame Robot-/Kamera-Configs nutzen möchtest,
kannst du hier gezielt die entsprechenden Klassen re-exportieren.
"""

__all__: list[str] = []