import tkinter as tk
import unittest
from unittest.mock import patch

from tokenwalk.app import Game


class WindowIconTests(unittest.TestCase):
    def test_game_installs_character_icon_and_keeps_it_through_restart(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with patch.object(root, "iconphoto", wraps=root.iconphoto) as register:
                game = Game(root, animate=False)
            self.assertTrue(
                hasattr(game, "window_icon"),
                "Game must retain its custom window icon",
            )
            image = game.window_icon
            register.assert_called_once_with(True, image)
            self.assertEqual((image.width(), image.height()), (64, 64))
            self.assertEqual(image.get(32, 20), (236, 231, 207))
            for action in (game.draw, game.restart, game.draw):
                action()
                root.update_idletasks()
                self.assertIs(game.window_icon, image)
                self.assertIn(str(image), root.tk.call("image", "names"))
        finally:
            root.destroy()
