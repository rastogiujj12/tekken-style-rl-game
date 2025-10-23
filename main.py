# main.py
import pygame, glob, imageio, random, torch, os, re, numpy as np
from pygame import mixer
from fighter import Fighter
from logger import Logger

mixer.init()
pygame.init()
PHASE = 2

PLAYER_1_MODEL_PATH = "weights/player_1/phase_1/model/_ep_"

if not PHASE ==1:
    player1_variants = [x for x in range(0,1000,50)]
    chosen_variant = random.choice(player1_variants)
print("chosen_variant", chosen_variant)

SAVE_INTERVAL = 50
TOTAL_EPISODES = 1000

# logger = Logger(log_dir="logs", filename_prefix=f"phase_{PHASE}")
step_logger = Logger(log_dir="logs", filename_prefix=f"phase_{PHASE}_steps")
episode_logger = Logger(log_dir="logs", filename_prefix=f"phase_{PHASE}_episodes")

print(f"[INFO] Logging to {step_logger.path()}")

#make folder structure
os.makedirs(f"weights/player_1/phase_{PHASE}/model", exist_ok=True)
os.makedirs(f"weights/player_1/phase_{PHASE}/optimizer", exist_ok=True)


os.makedirs(f"weights/player_2/phase_{PHASE}/model", exist_ok=True)
os.makedirs(f"weights/player_2/phase_{PHASE}/optimizer", exist_ok=True)

os.makedirs("recordings", exist_ok=True)

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
# pygame.mixer.music.load("assets/audio/music.mp3")
# pygame.mixer.music.set_volume(0.5)
# pygame.mixer.music.play(-1, 0.0, 5000)
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


def get_latest_episode(pattern):
    """
    Returns the latest episode number from a list of checkpoint files.
    """

    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No checkpoints found for pattern: {pattern}")
        return 0
    
    # Sort by episode number if available, otherwise by modified time
    def extract_episode(filename):
        match = re.search(r'ep_(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else -1

    files.sort(key=lambda f: extract_episode(f))
    latest_model_path = files[-1]
    ep_num = extract_episode(latest_model_path)
    if ep_num == -1:
        # fallback if file name doesn’t contain episode number
        ep_num = 0


    return ep_num

episodes_elapsed = get_latest_episode(f"weights/player_1/phase_{PHASE}/model/_ep_*.pth")
print("episodes_elapsed", episodes_elapsed)
if PHASE == 1:
    chosen_variant = episodes_elapsed
    # if episodes_elapsed >= TOTAL_EPISODES:
    #     PHASE = 2

# Create two deep-RL fighters
fighter_1 = Fighter(
    player=1,
    x=200, y=310, flip=False,
    data=WARRIOR_DATA, sprite_sheet=WARRIOR_SHEET, animation_steps=WARRIOR_STEPS,
    attack_sound=sword_fx, screen_width=SCREEN_WIDTH, 
    role="player", training_phase = PHASE, continue_from_episode = chosen_variant
)
fighter_2 = Fighter(
    player=2,
    x=700, y=310, flip=True,
    data=WIZARD_DATA, sprite_sheet=WIZARD_SHEET, animation_steps=WIZARD_STEPS,
    attack_sound=magic_fx, screen_width=SCREEN_WIDTH,
    role="enemy", training_phase = PHASE, continue_from_episode = 1050
)

if not PHASE == 1:
    fighter_1.epsilon = 0.0  # Fixed (opponent)
    fighter_2.epsilon_start = 1.0
    step_logger = Logger(log_dir="logs", filename_prefix=f"phase_{PHASE}_steps")
    episode_logger = Logger(log_dir="logs", filename_prefix=f"phase_{PHASE}_episodes")

    print(f"[INFO] Phase 2 logging started: {step_logger.path()}")

    q1 = loss1 = 0

run = True
episode_step=0
while run:
    if episodes_elapsed>TOTAL_EPISODES:
        run = False
    episode_step+=1
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
        fighter_1.move(fighter_2, round_over, elapsed)
        fighter_2.move(fighter_1, round_over, elapsed)

        if PHASE==1:
            loss1, q1 = fighter_1.optimize()
            loss2, q2 = fighter_2.optimize()
            if (loss1 is not None or loss2 is not None) and fighter_1.step_count % 500 == 0:
                step_logger.log(
                    episode=episodes_elapsed,
                    elapsed_time = elapsed,
                    step=fighter_1.step_count,
                    loss_p1=loss1 or 0.0,
                    loss_p2=loss2 or 0.0,
                    q_p1=q1 or 0.0,
                    q_p2=q2 or 0.0,
                    eps_p1=round(fighter_1.epsilon, 3),
                    eps_p2=round(fighter_2.epsilon, 3),
                    reward_p1=getattr(fighter_1, "episode_reward", 0.0),
                    reward_p2=getattr(fighter_2, "episode_reward", 0.0)
                )
        else:
            loss2, q2 = fighter_2.optimize()
            if loss2 is not None and fighter_1.step_count % 500 == 0:
                step_logger.log(
                    episode=episodes_elapsed,
                    episode_step = episode_step,
                    loss_p1=0.0,
                    loss_p2=loss2 or 0.0,
                    q_p1=0.0,
                    q_p2=q2 or 0.0,
                    eps_p1=round(fighter_1.epsilon, 3),
                    eps_p2=round(fighter_2.epsilon, 3),
                    reward_p1=getattr(fighter_1, "episode_reward", 0.0),
                    reward_p2=getattr(fighter_2, "episode_reward", 0.0),
                    opponent_ep =chosen_variant,
                )            

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
            
            # Save snapshots for player every N episodes
            if episodes_elapsed % SAVE_INTERVAL == 0:
                # if episodes_elapsed % 50 == 0:
                #     print(f"Ep {episodes_elapsed:4d} | ε1={fighter_1.epsilon:.3f} ε2={fighter_2.epsilon:.3f} | scores={score}")
                print(
                    f"Ep {episodes_elapsed:4d} | "
                    f"ε1={fighter_1.epsilon:.3f} ε2={fighter_2.epsilon:.3f} | "
                    f"avgQ1={q1 or 0:.3f} avgQ2={q2 or 0:.3f} | "
                    f"Loss1={loss1 or 0:.4f} Loss2={loss2 or 0:.4f}"
                )
                if PHASE==1:
                    torch.save(fighter_1.policy_net.state_dict(),
                            f"weights/player_1/phase_{PHASE}/model/_ep_{episodes_elapsed}.pth")
                    torch.save(fighter_1.optimizer.state_dict(), 
                            f"weights/player_1/phase_{PHASE}/optimizer/_ep_{episodes_elapsed}.pth")
                
                torch.save(fighter_2.policy_net.state_dict(),
                        f"weights/player_2/phase_{PHASE}/model/_ep_{episodes_elapsed}.pth")
                torch.save(fighter_2.optimizer.state_dict(), 
                            f"weights/player_2/phase_{PHASE}/optimizer/_ep_{episodes_elapsed}.pth")

            print(f"episode {episodes_elapsed} over, score: {score}")
            episodes_elapsed +=1

            fighter_1.current_episode = episodes_elapsed
            fighter_2.current_episode = episodes_elapsed
            fighter_1.anneal_epsilon()
            fighter_2.anneal_epsilon()

            episode_logger.log(
                episode=episodes_elapsed,
                epsilon_p1=round(fighter_1.epsilon, 3),
                epsilon_p2=round(fighter_2.epsilon, 3),
                score_p1=score[0],
                score_p2=score[1],
                opponent_ep = chosen_variant,
                episode_step = episode_step
            )
            
            # reset fighters
            if PHASE==2:
                chosen_variant = random.choice(player1_variants)
                checkpoint = torch.load(f"{PLAYER_1_MODEL_PATH}{chosen_variant}.pth", map_location=fighter_1.device)
                fighter_1.policy_net.load_state_dict(checkpoint)
                fighter_1.target_net.load_state_dict(checkpoint)
                print(f"[INFO] Loaded Fighter weights from {chosen_variant}")
            episode_step=0
            fighter_1.reset()
            fighter_2.reset()

    # if episodes_elapsed >= TOTAL_EPISODES and PHASE==1:
    #     episodes_elapsed = 0
    #     PHASE = 2

    #     fighter_1 = Fighter(
    #     player=1,
    #         x=200, y=310, flip=False,
    #         data=WARRIOR_DATA, sprite_sheet=WARRIOR_SHEET, animation_steps=WARRIOR_STEPS,
    #         attack_sound=sword_fx, screen_width=SCREEN_WIDTH, 
    #         role="player", training_phase = 2, continue_from_episode = TOTAL_EPISODES
    #     )
    #     fighter_2 = Fighter(
    #         player=2,
    #         x=700, y=310, flip=True,
    #         data=WIZARD_DATA, sprite_sheet=WIZARD_SHEET, animation_steps=WIZARD_STEPS,
    #         attack_sound=magic_fx, screen_width=SCREEN_WIDTH,
    #         role="enemy", training_phase = 2, continue_from_episode = TOTAL_EPISODES
    #     )

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            if episodes_elapsed>0:
                torch.save(fighter_1.policy_net.state_dict(),
                    f"weights/player_1/phase_{PHASE}/model/_ep_{episodes_elapsed}.pth")
                torch.save(fighter_1.optimizer.state_dict(), 
                    f"weights/player_1/phase_{PHASE}/optimizer/_ep_{episodes_elapsed}.pth")
                
                torch.save(fighter_2.policy_net.state_dict(),
                    f"weights/player_2/phase_{PHASE}/model/_ep_{episodes_elapsed}.pth")
                torch.save(fighter_2.optimizer.state_dict(), 
                    f"weights/player_2/phase_{PHASE}/optimizer/_ep_{episodes_elapsed}.pth")

            run = False

    pygame.display.update()

pygame.quit()
