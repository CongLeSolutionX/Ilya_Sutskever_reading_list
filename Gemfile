# Gemfile
source "https://rubygems.org" # The primary source for Ruby gems

# Specify your Ruby version (optional, but good practice if your host supports it)
# ruby "3.1.2" # Replace with your target Ruby version

# Core Jekyll gem
gem "jekyll", "~> 4.3" # Or your desired Jekyll version (e.g., "~> 3.9" for older sites)

# Common Jekyll plugins (add or remove based on your needs)
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.17"    # For generating an Atom feed
  gem "jekyll-sitemap", "~> 1.4"  # For generating a sitemap.xml
  gem "jekyll-seo-tag", "~> 2.8"  # For adding SEO metadata
  # gem "jekyll-paginate", "~> 1.1" # If you use pagination (jekyll-paginate-v2 is more modern)
  # gem "jekyll-gist", "~> 1.5"     # For embedding GitHub Gists
  # gem "jekyll-include-cache", "~> 0.2.1" # For caching includes to speed up builds
  # gem "webrick", "~> S1.7" # Required for `jekyll serve` on Ruby 3.0+ if not already present
end

# Add plugins for testing if you use them (e.g., HTMLProofer)
# group :development, :test do
#   gem "html-proofer", "~> 3.19"
# end

# Windows-specific gems if developing on Windows and encountering issues
# if Gem.win_platform?
#   gem "wdm", "~> 0.1.1"
# end