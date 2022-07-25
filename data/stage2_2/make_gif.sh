ffmpeg -i stage2.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" -loop 0 stage2.gif
