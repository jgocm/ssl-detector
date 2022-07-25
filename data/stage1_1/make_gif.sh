ffmpeg -i stage1.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" -loop 0 stage1.gif
