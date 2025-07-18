package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/mmcdole/gofeed"
)

// Story represents a Reddit news story
type Story struct {
	Title string
	Link  string
}

// SlackPayload defines the message format for Slack webhook
type SlackPayload struct {
	Text string `json:"text"`
}

// Constants
const (
	redditRSS    = "https://www.reddit.com/r/news/top/.rss?t=day"
	hfModelURL   = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
	summaryLimit = 5
)

func main() {
	// Load environment variables from .env for local development
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found — assuming environment variables are already set.")
	}

	// Get credentials from environment
	slackWebhook := os.Getenv("SLACK_WEBHOOK_URL")
	hfAPIKey := os.Getenv("HUGGINGFACE_API_KEY")

	if slackWebhook == "" || hfAPIKey == "" {
		log.Fatal("Missing SLACK_WEBHOOK_URL or HUGGINGFACE_API_KEY in environment")
	}

	// Fetch the top stories
	stories, err := fetchTopStories(summaryLimit)
	if err != nil {
		log.Fatalf("Failed to fetch stories: %v", err)
	}

	// Loop through each story and process
	for i, story := range stories {
		// Combine title and link for summarization input
		text := fmt.Sprintf("%s - %s", story.Title, story.Link)

		// Summarize the story
		summary, err := summarizeWithHuggingFace(hfAPIKey, text)
		if err != nil {
			log.Printf("Error summarizing story %d: %v", i+1, err)
			continue
		}

		// Build Slack message (without hyperlink)
		message := fmt.Sprintf(
			"📌 *Story %d*\n*Title:* %s\n*Summary:* %s\n\n---",
			i+1,
			story.Title,
			summary,
		)

		// Send to Slack
		err = postToSlack(slackWebhook, message)
		if err != nil {
			log.Printf("Error posting to Slack: %v", err)
		}
	}
}

// fetchTopStories fetches the top N Reddit news stories from the RSS feed
func fetchTopStories(limit int) ([]Story, error) {
	fp := gofeed.NewParser()
	feed, err := fp.ParseURL(redditRSS)
	if err != nil {
		return nil, err
	}

	var stories []Story
	for i, item := range feed.Items {
		if i >= limit {
			break
		}
		stories = append(stories, Story{
			Title: item.Title,
			Link:  item.Link,
		})
	}
	return stories, nil
}

// summarizeWithHuggingFace sends the given text to Hugging Face and returns a summary
func summarizeWithHuggingFace(apiKey, text string) (string, error) {
	body, _ := json.Marshal(map[string]string{"inputs": text})

	req, err := http.NewRequest("POST", hfModelURL, bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 40 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result []map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if len(result) > 0 && result[0]["summary_text"] != "" {
		return result[0]["summary_text"], nil
	}

	return "Summary unavailable", nil
}

// postToSlack sends the given message to the specified Slack webhook URL
func postToSlack(webhookURL, message string) error {
	payload := SlackPayload{Text: message}
	data, _ := json.Marshal(payload)

	resp, err := http.Post(webhookURL, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Slack responded with status: %v", resp.Status)
	}
	return nil
}
