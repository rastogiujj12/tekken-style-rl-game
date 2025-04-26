# main.py
import pygame
from pygame import mixer
from fighter import Fighter

mixer.init()
pygame.init()

# create game window
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
GROUND_HEIGHT = SCREEN_HEIGHT - 293

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Brawler")

clock = pygame.time.Clock()
FPS = 60

# colors
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# game variables
intro_count = 3
last_count_update = pygame.time.get_ticks()
score = [0, 0]
round_over = False
ROUND_OVER_COOLDOWN = 2000
round_time_limit = 90
round_start_time = pygame.time.get_ticks()

# fighter setup data (unchanged)
WARRIOR_DATA = [162, 4, [72, 56]]
WIZARD_DATA  = [250, 3, [112, 107]]
WARRIOR_SHEET = pygame.image.load("assets/images/warrior/Sprites/warrior.png").convert_alpha()
WIZARD_SHEET  = pygame.image.load("assets/images/wizard/Sprites/wizard.png").convert_alpha()
WARRIOR_STEPS = [10, 8, 1, 7, 7, 3, 7]
WIZARD_STEPS  = [8,  8, 1, 8, 8, 3, 7]

# audio
pygame.mixer.music.load("assets/audio/music.mp3")
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(-1, 0.0, 5000)
sword_fx = pygame.mixer.Sound("assets/audio/sword.wav");   sword_fx.set_volume(0.5)
magic_fx = pygame.mixer.Sound("assets/audio/magic.wav");   magic_fx.set_volume(0.75)

# graphics
bg = pygame.image.load("assets/images/background/background.jpg").convert_alpha()
victory_img = pygame.image.load("assets/images/icons/victory.png").convert_alpha()

# fonts
count_font = pygame.font.Font("assets/fonts/turok.ttf", 80)
score_font = pygame.font.Font("assets/fonts/turok.ttf", 30)

def draw_text(text, font, color, x, y):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

def draw_bg():
    screen.blit(pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT)), (0, 0))

def draw_health_bar(h, x, y):
    ratio = h / 100
    pygame.draw.rect(screen, WHITE,  (x-2, y-2, 404, 34))
    pygame.draw.rect(screen, RED,    (x,   y,   400, 30))
    pygame.draw.rect(screen, YELLOW, (x,   y,   400 * ratio, 30))


# Create two deep-RL fighters
fighter_1 = Fighter(
    player=1,
    x=200, y=310, flip=False,
    data=WARRIOR_DATA, sprite_sheet=WARRIOR_SHEET, animation_steps=WARRIOR_STEPS,
    attack_sound=sword_fx, screen_width=SCREEN_WIDTH
)
fighter_2 = Fighter(
    player=2,
    x=700, y=310, flip=True,
    data=WIZARD_DATA, sprite_sheet=WIZARD_SHEET, animation_steps=WIZARD_STEPS,
    attack_sound=magic_fx, screen_width=SCREEN_WIDTH
)

run = True
while run:
    clock.tick(FPS)
    draw_bg()
    draw_health_bar(fighter_1.health, 20, 20)
    draw_health_bar(fighter_2.health, 580, 20)
    draw_text(f"P1: {score[0]}", score_font, RED, 20, 60)
    draw_text(f"P2: {score[1]}", score_font, RED, 580, 60)

    if intro_count > 0:
        draw_text(str(intro_count), count_font, RED, SCREEN_WIDTH/2-20, SCREEN_HEIGHT/3)
        if pygame.time.get_ticks() - last_count_update >= 1000:
            intro_count -= 1
            last_count_update = pygame.time.get_ticks()
            if intro_count == 0:
                round_start_time = pygame.time.get_ticks()
    else:
        elapsed = (pygame.time.get_ticks() - round_start_time)/1000
        rem    = round_time_limit - elapsed
        mins   = int(rem)//60
        secs   = int(rem)%60
        draw_text(f"{mins}:{secs:02d}", count_font, RED, SCREEN_WIDTH/2-40, 10)

        # agents move, learn, and draw themselves
        fighter_1.move(fighter_2, round_over)
        fighter_2.move(fighter_1, round_over)
        fighter_1.update()
        fighter_2.update()
        fighter_1.draw(screen)
        fighter_2.draw(screen)

        if not round_over:
            if not fighter_1.alive:
                score[1] += 1
                round_over = True
                round_over_time = pygame.time.get_ticks()
            elif not fighter_2.alive:
                score[0] += 1
                round_over = True
                round_over_time = pygame.time.get_ticks()

        if rem <= 0 or (round_over and pygame.time.get_ticks()-round_over_time > ROUND_OVER_COOLDOWN):
            round_over = False
            intro_count = 3
            fighter_1.reset()
            fighter_2.reset()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()
