# Pathfinding Showdown - Printf’s Sneaky Probe

This is a **StarCraft 2 AI Challenge** where you must programmatically guide a Protoss **Probe** to a designated location using custom pathfinding logic. The challenge is based on **SC2MapAnalysis**, which provides grid-based pathing solutions for AI bots.

---

## **Setup Instructions**

### **1. Clone This Repository**
```bash
git clone https://github.com/YOUR-USERNAME/Pathfinding-Showdown.git
cd Pathfinding-Showdown
```

### **2. Install Dependencies**
You will need Python 3.8+ and `poetry` to manage dependencies.

```bash
pip install poetry  # If you don't have it installed already
poetry install
```

### **3. Install StarCraft 2 and Maps**
Ensure you have **StarCraft 2** installed. If you don't have it, download it from Blizzard’s website.

#### **On Windows**
```powershell
cd "C:\Program Files (x86)\StarCraft II"
```

#### **On Linux (Lutris or WINE SC2 Install)**
```bash
cd ~/.wine/drive_c/Program\ Files\ \(x86\)/StarCraft\ II/
```

Then, **download the map** (if needed) and place it in the appropriate SC2 maps folder.

---

## **Getting Started**

### **1. Run the Challenge**
To start the challenge, use:
```bash
python run.py --Map LightShade_Pathing_0
```
This will launch StarCraft 2 with your bot running against the challenge scenario.

### **2. Modify Your Bot (Implementation Needed)**
Your bot is in `bot.py`. You'll need to implement **pathfinding logic** instead of issuing direct movement commands.

Key areas to work on:
- Implementing pathfinding using **SC2MapAnalysis**.
- Moving the **probe step-by-step** rather than `unit.move(goal)`.
- Avoiding obstacles and taking the shortest path possible.

### **3. Debugging and Visualization**
You can use **debugging tools** to visualize your probe's movement:
```python
self.client.debug_box_out(position, color=color)
```
Use this to **track the path** and adjust your algorithm accordingly.

---

## **Required Plugins**

This challenge requires the **SC2MapAnalysis** plugin for pathfinding. Install it by cloning the repository:
```bash
git clone https://github.com/spudde123/SC2MapAnalysis.git
```
Then, place the `map_analyzer` folder inside your project directory.

For detailed documentation on **SC2MapAnalysis**, refer to:
[SC2MapAnalysis Docs](https://eladyaniv01.github.io/SC2MapAnalysis/index.html)

---

## **Reference**
This project is based on the **VersusAI SC2 Bot Template**. You can refer to its original README for general setup instructions:
[VersusAI SC2 Bot Template](https://github.com/Vers-AI/versusai-sc2-bot-template)

---

## **Next Steps**
1. Implement your pathfinding logic.
2. Test and visualize the probe's movement.
3. Improve your bot's efficiency using the `on_end` output.
4. Compare results with other implementations!

Good luck, and may your probe find the optimal path! 🚀

